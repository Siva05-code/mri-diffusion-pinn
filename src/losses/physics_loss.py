"""
Physics-Informed MRI Reconstruction Loss.

Combines:
1. K-space data consistency (fidelity to measured physical k-space):
   L_data_consistency = || M ⊙ (A(I_pred; φ) - y_measured) ||_1 / (|| M ⊙ y_measured ||_1 + eps)
2. Image-gradient smoothness (physical edge-preserving total variation regularization):
   L_gradient = ||∇_x I_pred||_1 + ||∇_y I_pred||_1

Combined:
   L_physics = L_data_consistency + lambda_gradient * L_gradient
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

from src.losses.kspace_loss import KSpaceDataConsistencyLoss


class ImageGradientLoss(nn.Module):
    """
    Computes L1 norm of spatial image gradients (Total Variation regularization).
        L_gradient = ||∇_x I||_1 + ||∇_y I||_1

    Uses forward finite differences to preserve anatomical edges while penalizing
    unphysical high-frequency reconstruction artifacts.
    """
    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Image tensor [B, 1, H, W] or [B, H, W].
        Returns:
            torch.Tensor: Scalar gradient loss.
        """
        if img.ndim == 3:
            img = img.unsqueeze(1)
        elif img.ndim == 2:
            img = img.unsqueeze(0).unsqueeze(0)

        # Finite differences along vertical (y) and horizontal (x) axes
        diff_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        diff_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])

        return torch.mean(diff_y) + torch.mean(diff_x)


class MRIPhysicsLoss(nn.Module):
    """
    Composite MRI Physics Loss:
        L_physics = L_data_consistency + lambda_gradient * L_gradient

    Args:
        lambda_gradient (float): Weight for gradient smoothness regularization (default: 0.1).
        relative (bool): Whether to use relative normalization for data consistency (default: True).
        include_data_consistency (bool): Whether to include data consistency inside physics loss (default: True).
    """
    def __init__(
        self,
        lambda_gradient: float = 0.1,
        relative: bool = True,
        include_data_consistency: bool = True,
    ):
        super().__init__()
        self.lambda_gradient = lambda_gradient
        self.include_data_consistency = include_data_consistency
        self.kspace_fidelity = KSpaceDataConsistencyLoss(norm="l1", relative=relative)
        self.gradient_smoothness = ImageGradientLoss()

    def forward(
        self,
        i_pred: torch.Tensor,
        y_measured: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics loss components.

        Args:
            i_pred: Predicted image [B, 1, H, W].
            y_measured: Optional acquired k-space measurements.
            mask: Optional binary sampling mask.
            phase: Optional spatial phase tensor.

        Returns:
            Dict containing:
                - 'physics_loss': Total weighted physics loss
                - 'data_consistency_loss': K-space data consistency term
                - 'gradient_loss': Image-gradient smoothness term
        """
        grad_loss = self.gradient_smoothness(i_pred)

        if self.include_data_consistency and y_measured is not None and mask is not None:
            dc_loss = self.kspace_fidelity(i_pred, y_measured, mask, phase=phase)
            total_physics = dc_loss + self.lambda_gradient * grad_loss
        else:
            dc_loss = torch.tensor(0.0, device=i_pred.device)
            total_physics = self.lambda_gradient * grad_loss

        return {
            "physics_loss": total_physics,
            "data_consistency_loss": dc_loss,
            "gradient_loss": grad_loss,
        }
