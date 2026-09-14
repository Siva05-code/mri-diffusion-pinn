"""
Modular Total Reconstruction Loss for Accelerated MRI.

Combines image-domain reconstruction loss with optional physics-based k-space data-consistency:
    Loss = lambda_image * L_image + lambda_kspace * L_kspace

For standard U-Net baseline, lambda_kspace is set to 0.0.
"""

from typing import Optional, Dict
import torch
import torch.nn as nn

from src.losses.image_loss import build_image_loss
from src.losses.kspace_loss import KSpaceDataConsistencyLoss


class TotalReconstructionLoss(nn.Module):
    """
    Modular reconstruction loss wrapper.

    Args:
        image_loss_type (str): 'l1' or 'mse'.
        lambda_image (float): Weight for image-domain loss (default: 1.0).
        lambda_kspace (float): Weight for k-space loss (default: 0.0 for pure baseline).
    """
    def __init__(
        self,
        image_loss_type: str = "l1",
        lambda_image: float = 1.0,
        lambda_kspace: float = 0.0,
    ):
        super().__init__()
        self.lambda_image = lambda_image
        self.lambda_kspace = lambda_kspace

        self.image_loss_fn = build_image_loss(image_loss_type)
        if self.lambda_kspace > 0.0:
            self.kspace_loss_fn = KSpaceDataConsistencyLoss(norm="l1", relative=True)
        else:
            self.kspace_loss_fn = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        y_measured: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted total loss.

        Returns:
            dict containing:
                - 'loss': scalar total loss
                - 'image_loss': image-domain loss
                - 'kspace_loss': k-space loss (0.0 if disabled)
        """
        img_loss = self.image_loss_fn(pred, target)
        total_loss = self.lambda_image * img_loss
        kspace_loss = torch.tensor(0.0, device=pred.device)

        if self.lambda_kspace > 0.0 and self.kspace_loss_fn is not None:
            if y_measured is not None and mask is not None:
                kspace_loss = self.kspace_loss_fn(pred, y_measured, mask)
                total_loss = total_loss + self.lambda_kspace * kspace_loss

        return {
            "loss": total_loss,
            "image_loss": img_loss,
            "kspace_loss": kspace_loss,
        }

