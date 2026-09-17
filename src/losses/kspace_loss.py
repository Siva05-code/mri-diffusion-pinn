"""
K-Space Domain Data-Consistency Loss for Accelerated MRI.

Measures the discrepancy between reference k-space measurements and
the forward Fourier transform of the reconstructed image at sampled trajectory points:
    L_kspace = || M ⊙ (A(I_pred; φ) - y_meas) ||_1 / (|| M ⊙ y_meas ||_1 + eps)

Correctly handles complex k-space using differentiable complex modulus:
    |z| = sqrt(Re(z)^2 + Im(z)^2 + eps)
"""

from typing import Optional
import torch
import torch.nn as nn
from src.physics.forward_operator import MRIForwardOperator
from src.physics.fft import image_to_kspace_torch


class KSpaceDataConsistencyLoss(nn.Module):
    """
    K-Space Loss comparing forward model of predicted reconstruction against reference k-space at sampled locations.

    Args:
        norm (str): 'l1' or 'l2' norm (default: 'l1').
        relative (bool): Whether to normalize by the norm of reference k-space at sampled locations (default: True).
        eps (float): Numerical stability epsilon.
    """
    def __init__(self, norm: str = "l1", relative: bool = True, eps: float = 1e-12):
        super().__init__()
        self.norm = norm.lower()
        self.relative = relative
        self.eps = eps
        self.forward_op = MRIForwardOperator()

    def forward(
        self,
        i_pred: torch.Tensor,
        k_gt: torch.Tensor,
        mask: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute k-space loss at sampled locations:
            L_kspace = || M * (A(I_pred) - K_meas) ||_1

        Args:
            i_pred: Predicted image [B, 1, H, W], [B, H, W], or [H, W].
            k_gt: Measured/reference k-space (complex tensor).
            mask: Binary sampling mask (1 for measured lines, 0 unmeasured).
            phase: Optional spatial phase tensor.

        Returns:
            torch.Tensor: Differentiable scalar k-space loss.
        """
        if i_pred.ndim == 4 and i_pred.shape[1] == 1:
            img = i_pred.squeeze(1)
            phase_in = phase.squeeze(1) if (phase is not None and phase.ndim == 4 and phase.shape[1] == 1) else phase
        else:
            img = i_pred
            phase_in = phase

        if not torch.is_complex(k_gt):
            if k_gt.shape[-1] == 2:
                k_gt = torch.complex(k_gt[..., 0], k_gt[..., 1])
            else:
                k_gt = k_gt.to(torch.complex64)

        img_h, img_w = img.shape[-2], img.shape[-1]
        k_h, k_w = k_gt.shape[-2], k_gt.shape[-1]

        # Handle size matching between image and k-space via forward operator
        if (img_h, img_w) == (k_h, k_w):
            if not torch.is_complex(img):
                if phase_in is not None:
                    img_c = img.to(torch.complex64) * torch.exp(1j * phase_in.to(img.device))
                else:
                    img_c = img.to(torch.complex64)
            else:
                img_c = img
            k_pred = image_to_kspace_torch(img_c)
        else:
            k_pred = self.forward_op.forward_kspace(img, phase=phase_in, target_shape=(k_h, k_w))

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
