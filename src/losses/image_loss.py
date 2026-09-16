"""
Image-Domain Reconstruction Loss Functions for MRI.

Provides modular L1 and MSE losses for training image reconstruction models.
Provides differentiable L1 and MSE losses for training image reconstruction models:
    L_image = ||I_pred - I_gt||_1
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1ImageLoss(nn.Module):
    """
    Mean Absolute Error (L1) loss in the image domain.
    Measures the pixel-wise absolute difference between reconstructed and reference images.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 loss between prediction and ground truth.
        Supports shapes [B, C, H, W], [B, H, W], or [H, W].
        """
        # Ensure dimensions match
        if pred.ndim != target.ndim:
            if pred.ndim == 4 and target.ndim == 3:
                target = target.unsqueeze(1)
            elif pred.ndim == 3 and target.ndim == 4:
                pred = pred.unsqueeze(1)
            elif pred.ndim == 4 and target.ndim == 2:
                target = target.unsqueeze(0).unsqueeze(0)

        return self.criterion(pred, target)


class MSEImageLoss(nn.Module):
    """
    Mean Squared Error (MSE / L2) loss in the image domain.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != target.ndim:
            if pred.ndim == 4 and target.ndim == 3:
                target = target.unsqueeze(1)
            elif pred.ndim == 3 and target.ndim == 4:
                pred = pred.unsqueeze(1)
            elif pred.ndim == 4 and target.ndim == 2:
                target = target.unsqueeze(0).unsqueeze(0)

        return self.criterion(pred, target)


def build_image_loss(loss_type: str = "l1") -> nn.Module:
    """Factory function for image-domain losses."""
    loss_type = loss_type.lower()
    if loss_type == "l1":
        return L1ImageLoss()
    elif loss_type in ("l2", "mse"):
        return MSEImageLoss()
    else:
        raise ValueError(f"Unknown image loss type: {loss_type}. Supported: 'l1', 'mse'")

