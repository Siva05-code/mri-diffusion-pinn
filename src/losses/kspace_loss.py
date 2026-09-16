"""
K-Space Domain Data-Consistency Loss for Accelerated MRI.

Measures the discrepancy between reference k-space measurements and
the forward Fourier transform of the reconstructed image at sampled trajectory points:
    L_kspace = || M ⊙ (FFT(I_pred) - K_gt) ||_1 / (|| M ⊙ K_gt ||_1 + eps)

Correctly handles complex k-space using complex modulus:
    |z| = sqrt(Re(z)^2 + Im(z)^2 + eps)
"""

from typing import Optional
import torch
import torch.nn as nn
from src.physics.fft import image_to_kspace_torch


class KSpaceDataConsistencyLoss(nn.Module):
    """
    K-Space Loss comparing predicted k-space against reference k-space at sampled locations.

    Args:
        norm (str): 'l1' or 'l2' norm.
        relative (bool): Whether to normalize by the norm of reference k-space at sampled locations (default: True).
        eps (float): Numerical stability epsilon.
    """
    def __init__(self, norm: str = "l1", relative: bool = True, eps: float = 1e-12):
        super().__init__()
        self.norm = norm.lower()
        self.relative = relative
        self.eps = eps

    def forward(
        self,
        i_pred: torch.Tensor,
        k_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute k-space loss at sampled locations:
            L_kspace = || M * (FFT(I_pred) - K_gt) ||_1

        Args:
            i_pred: Predicted image [B, 1, H, W], [B, H, W], or [H, W].
            k_gt: Reference k-space (complex tensor).
            mask: Binary sampling mask (1 for measured lines, 0 unmeasured).

        Returns:
            torch.Tensor: Differentiable scalar k-space loss.
        """
        # Ensure i_pred is complex [..., H, W]
        if i_pred.ndim == 4 and i_pred.shape[1] == 1:
            img = i_pred.squeeze(1)
        else:
            img = i_pred

        if not torch.is_complex(img):
            img_complex = img.to(torch.complex64)
        else:
            img_complex = img

        if not torch.is_complex(k_gt):
            if k_gt.shape[-1] == 2:
                k_gt = torch.complex(k_gt[..., 0], k_gt[..., 1])
            else:
                k_gt = k_gt.to(torch.complex64)

        img_h, img_w = img_complex.shape[-2], img_complex.shape[-1]
        k_h, k_w = k_gt.shape[-2], k_gt.shape[-1]

        # Handle size matching between image and k-space
        if (img_h, img_w) == (k_h, k_w):
            k_pred = image_to_kspace_torch(img_complex)
        else:
            pad_top = (k_h - img_h) // 2
            pad_bottom = k_h - img_h - pad_top
            pad_left = (k_w - img_w) // 2
            pad_right = k_w - img_w - pad_left
            img_padded = torch.nn.functional.pad(
                img_complex, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
            )
            k_pred = image_to_kspace_torch(img_padded)

        mask = mask.to(k_pred.device, dtype=k_pred.real.dtype)

        # Difference at sampled locations only
        diff = mask * (k_pred - k_gt)

        # Differentiable complex magnitude: sqrt(Re^2 + Im^2 + eps)
        diff_mag = torch.sqrt(diff.real**2 + diff.imag**2 + self.eps)
        mask_sum = torch.sum(mask) + self.eps

        if self.norm == "l1":
            loss = torch.sum(diff_mag) / mask_sum
            if self.relative:
                ref_mag = torch.sqrt((mask * k_gt).real**2 + (mask * k_gt).imag**2 + self.eps)
                denom = (torch.sum(ref_mag) / mask_sum) + self.eps
                loss = loss / denom
        elif self.norm == "l2":
            diff_sq = diff.real**2 + diff.imag**2
            loss = torch.sum(diff_sq) / mask_sum
            if self.relative:
                ref_sq = (mask * k_gt).real**2 + (mask * k_gt).imag**2
                denom = (torch.sum(ref_sq) / mask_sum) + self.eps
                loss = loss / denom
        else:
            raise ValueError(f"Unsupported norm: {self.norm}. Supported: 'l1', 'l2'")

        return loss
