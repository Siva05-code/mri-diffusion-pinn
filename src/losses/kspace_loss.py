"""
K-Space Domain Data-Consistency Loss for MRI.

Measures the discrepancy between the sampled acquired k-space measurements and
the forward Fourier transform of the reconstructed image at sampled trajectory points:
    L_kspace = || Mask * (FFT(x_pred) - y_measured) ||_1 / || Mask * y_measured ||_1
"""

from typing import Optional
import torch
import torch.nn as nn
from src.physics.fft import image_to_kspace_torch


class KSpaceDataConsistencyLoss(nn.Module):
    """
    K-Space Data Consistency Loss.

    Args:
        norm (str): 'l1' or 'l2' norm.
        relative (bool): Whether to normalize by the norm of y_measured.
    """
    def __init__(self, norm: str = "l1", relative: bool = True):
        super().__init__()
        self.norm = norm.lower()
        self.relative = relative

    def forward(
        self,
        x_pred: torch.Tensor,
        y_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute k-space data consistency loss.

        Args:
            x_pred: Predicted magnitude/complex image [B, 1, H, W] or [B, H, W].
            y_measured: Acquired undersampled k-space tensor (complex).
            mask: Binary sampling mask (same broadcastable shape).

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        # Ensure x_pred has appropriate shape and complex representation
        if x_pred.ndim == 4 and x_pred.shape[1] == 1:
            x_pred = x_pred.squeeze(1)

        if not torch.is_complex(x_pred):
            x_pred_complex = x_pred.to(torch.complex64)
        else:
            x_pred_complex = x_pred

        # Forward FFT to k-space domain
        y_pred = image_to_kspace_torch(x_pred_complex)

        # Apply sampling mask to isolate acquired lines
        diff = mask * (y_pred - y_measured)

        if self.norm == "l1":
            loss = torch.mean(torch.abs(diff))
            if self.relative:
                denom = torch.mean(torch.abs(mask * y_measured)) + 1e-12
                loss = loss / denom
        elif self.norm == "l2":
            loss = torch.mean(torch.abs(diff) ** 2)
            if self.relative:
                denom = torch.mean(torch.abs(mask * y_measured) ** 2) + 1e-12
                loss = loss / denom
        else:
            raise ValueError(f"Unsupported norm: {self.norm}. Use 'l1' or 'l2'.")

        return loss

