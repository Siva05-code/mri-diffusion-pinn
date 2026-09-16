"""
Composite Reconstruction Loss for Diffusion + PINN Accelerated MRI.

Implements the multi-term objective:
    L_total = lambda_image * L_image + alpha * L_kspace + beta * L_physics

where:
    L_image   = ||I_pred - I_gt||_1
    L_kspace  = ||M ⊙ (FFT(I_pred) - K_gt)||_1 / (||M ⊙ K_gt||_1 + eps)
    L_physics = lambda_gradient * ||∇I_pred||_1  (+ optional L_data_consistency if alpha=0)

Design Note:
    To prevent double-counting k-space fidelity when both alpha (lambda_kspace) and
    beta (lambda_physics) are active, L_kspace handles Fourier trajectory fidelity,
    while L_physics enforces the image-gradient smoothness prior. If lambda_kspace is 0,
    L_physics can include data consistency for ablation C.
"""

from typing import Optional, Dict
import torch
import torch.nn as nn

from src.losses.image_loss import build_image_loss
from src.losses.kspace_loss import KSpaceDataConsistencyLoss
from src.losses.physics_loss import MRIPhysicsLoss, ImageGradientLoss


class TotalReconstructionLoss(nn.Module):
    """
    Composite loss combining image fidelity, k-space fidelity, and MRI physics.

    Args:
        image_loss_type (str): 'l1' or 'mse' (default: 'l1').
        lambda_image (float): Weight for image loss (default: 1.0).
        lambda_kspace (float): Weight alpha for k-space loss (default: 1.0).
        lambda_physics (float): Weight beta for physics loss (default: 1.0).
        lambda_gradient (float): Smoothness regularization weight inside physics loss (default: 0.1).
        relative_kspace (bool): Whether to use relative normalization for k-space loss (default: True).
    """
    def __init__(
        self,
        image_loss_type: str = "l1",
        lambda_image: float = 1.0,
        lambda_kspace: float = 1.0,
        lambda_physics: float = 1.0,
        lambda_gradient: float = 0.1,
        relative_kspace: bool = True,
    ):
        super().__init__()
        self.lambda_image = lambda_image
        self.lambda_kspace = lambda_kspace
        self.lambda_physics = lambda_physics
        self.lambda_gradient = lambda_gradient

        self.image_loss_fn = build_image_loss(image_loss_type)
        self.kspace_loss_fn = KSpaceDataConsistencyLoss(norm="l1", relative=relative_kspace)
        self.gradient_loss_fn = ImageGradientLoss()

    def forward(
        self,
        i_pred: torch.Tensor,
        i_gt: torch.Tensor,
        y_measured: Optional[torch.Tensor] = None,
        k_gt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total composite loss and component terms.

        Args:
            i_pred: Reconstructed image [B, 1, H, W].
            i_gt: Ground-truth reference image [B, 1, H, W].
            y_measured: Undersampled acquired k-space measurements.
            k_gt: Full/reference k-space (if available, else y_measured is used).
            mask: Binary sampling mask.

        Returns:
            Dict containing:
                - 'total_loss': Weighted composite loss
                - 'image_loss': Image-domain loss
                - 'kspace_loss': K-space domain loss
                - 'physics_loss': MRI physics loss
                - 'gradient_loss': Spatial gradient TV loss
        """
        # 1. Image loss
        image_loss = self.image_loss_fn(i_pred, i_gt)
        total = self.lambda_image * image_loss

        # 2. K-space loss (Fourier fidelity)
        ref_kspace = k_gt if k_gt is not None else y_measured
        if ref_kspace is not None and mask is not None:
            kspace_loss = self.kspace_loss_fn(i_pred, ref_kspace, mask)
        else:
            kspace_loss = torch.tensor(0.0, device=i_pred.device)

        if self.lambda_kspace > 0.0:
            total = total + self.lambda_kspace * kspace_loss

        # 3. Physics loss (Image-gradient smoothness prior + data consistency if alpha=0)
        gradient_loss = self.gradient_loss_fn(i_pred)

        # If lambda_kspace is 0, include kspace_loss inside physics loss (Ablation C)
        # Otherwise, avoid double counting: physics loss is the gradient smoothness prior
        if self.lambda_kspace == 0.0 and ref_kspace is not None and mask is not None:
            physics_loss = kspace_loss + self.lambda_gradient * gradient_loss
        else:
            physics_loss = self.lambda_gradient * gradient_loss

        if self.lambda_physics > 0.0:
            total = total + self.lambda_physics * physics_loss

        return {
            "loss": total,
            "total_loss": total,
            "image_loss": image_loss,
            "kspace_loss": kspace_loss,
            "physics_loss": physics_loss,
            "gradient_loss": gradient_loss,
        }
