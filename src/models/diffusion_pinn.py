"""
Unified Diffusion + PINN Architecture for Accelerated MRI Reconstruction.

Pipeline:
    Undersampled k-space
            ↓
     Zero-filled image
            ↓
  ┌─────────────────────┐
  │ Conditional Diffusion│
  │ Pre-Reconstruction   │
  └──────────┬──────────┘
             ↓
      Coarse image
             ↓
  ┌─────────────────────┐
  │ PINN Refinement      │
  │ Physics-informed     │
  └──────────┬──────────┘
             ↓
      Final image

Outputs both (coarse, refined) reconstructions to support modular benchmarking
and scientific ablation analysis against the diffusion-only baseline.
"""

from typing import Tuple, Dict, Optional, Union
import torch
import torch.nn as nn

from src.models.diffusion import ConditionalDiffusionUNet
from src.models.diffusion_schedule import DiffusionSchedule
from src.models.pinn import PINNRefinementNet
from src.physics.data_consistency import apply_data_consistency


class DiffusionPINN(nn.Module):
    """
    Joint Diffusion Pre-Reconstruction + PINN Refinement Framework.

    Args:
        diffusion_base_channels: Capacity of the conditional diffusion U-Net (default: 32).
        diffusion_time_dim: Dimension of sinusoidal time embedding (default: 128).
        pinn_base_channels: Capacity of the PINN refinement network (default: 32).
        pinn_num_blocks: Number of residual blocks in the PINN refinement net (default: 4).
        timesteps: Total number of diffusion timesteps (default: 100).
        enforce_dc: Whether to apply explicit data consistency to the final output (default: True).
    """
    def __init__(
        self,
        diffusion_base_channels: int = 32,
        diffusion_time_dim: int = 128,
        pinn_base_channels: int = 32,
        pinn_num_blocks: int = 4,
        timesteps: int = 100,
        enforce_dc: bool = True,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.enforce_dc = enforce_dc

        # 1. Conditional Diffusion Pre-Reconstruction Network
        self.diffusion = ConditionalDiffusionUNet(
            base_channels=diffusion_base_channels,
            time_dim=diffusion_time_dim,
        )

        # Diffusion noise schedule
        self.schedule = DiffusionSchedule(timesteps=timesteps)

        # 2. PINN Refinement Network (accepts coarse + zero-filled condition)
        self.pinn = PINNRefinementNet(
            in_channels=2,
            out_channels=1,
            base_channels=pinn_base_channels,
            num_blocks=pinn_num_blocks,
        )

    def to(self, device):
        self.schedule.to(device)
        return super().to(device)

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        pred_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable analytical clean image estimation via Tweedie's formula:
            x_0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)
        """
        alpha_bar = self.schedule.alpha_bars[t].to(x_t.device)
        alpha_bar = alpha_bar[:, None, None, None]
        return self.schedule.predict_x0(x_t, pred_noise, t, clamp=True)

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        pred_x0 = (x_t - sqrt_one_minus_alpha_bar * pred_noise) / (sqrt_alpha_bar + 1e-8)
        return torch.clamp(pred_x0, 0.0, 1.0)

    def forward_train(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        y_measured: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-end differentiable forward pass for joint training.

        Flows gradients back through PINN into the Diffusion network:
        1. Adds noise to target to create x_t.
        Training-time diffusion denoising path:
        1. Adds noise to target to create x_t at timestep t.
        2. Diffusion predicts epsilon.
        3. Differentiably computes coarse x_0 estimate via Tweedie's formula.
        4. Passes coarse x_0 and zero-filled condition through PINN refinement.
        5. Optionally applies k-space data consistency.

        Flows gradients back through PINN into the Diffusion network.

        Returns:
            Dict containing:
                - 'coarse': Coarse image from diffusion pre-reconstruction [B, 1, H, W]
                - 'refined': Refined image from PINN [B, 1, H, W]
                - 'pred_noise': Predicted noise from diffusion U-Net
                - 'noise': Ground truth injected Gaussian noise
                - 't': Timestep indices
        """
        # Ensure strictly 4D [B, 1, H, W]
        while target.ndim > 4:
            target = target.squeeze(1)
        while condition.ndim > 4:
            condition = condition.squeeze(1)
        if target.ndim == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 3:
            target = target.unsqueeze(1)
        if condition.ndim == 2:
            condition = condition.unsqueeze(0).unsqueeze(0)
        elif condition.ndim == 3:
            condition = condition.unsqueeze(1)

        device = target.device

        if t is None:
            t = torch.randint(0, self.timesteps, (target.shape[0],), device=device)

        if noise is None:
            noise = torch.randn_like(target)

        # 1. Forward diffusion process
        x_t = self.schedule.add_noise(target, noise, t)

        # 2. Diffusion model predicts noise
        pred_noise = self.diffusion(x_t, condition, t)

        # 3. Differentiable coarse reconstruction
        coarse = self.predict_x0_from_noise(x_t, pred_noise, t)

        # 4. PINN Refinement
        refined = self.pinn(coarse=coarse, zero_filled=condition)

        # 5. Optional explicit data consistency projection
        if self.enforce_dc and y_measured is not None and mask is not None:
            refined = apply_data_consistency(
                x_image=refined,
                k_measured=y_measured,
                mask=mask,
                return_complex=False,
            )
            refined = torch.clamp(refined, 0.0, 1.0)

        return {
            "coarse": coarse,
            "refined": refined,
            "pred_noise": pred_noise,
            "noise": noise,
            "t": t,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        y_measured: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference sampling pipeline:
        1. Full reverse diffusion sampling -> coarse pre-reconstruction
        2. PINN refinement -> refined reconstruction
        3. Data consistency projection (if enabled)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (coarse, refined)
        """
        while condition.ndim > 4:
            condition = condition.squeeze(1)
        if condition.ndim == 2:
            condition = condition.unsqueeze(0).unsqueeze(0)
        elif condition.ndim == 3:
            condition = condition.unsqueeze(1)

        if device is None:
            device = condition.device

        self.eval()

        # Step 1: Coarse reconstruction via reverse diffusion
        coarse = self.schedule.sample(
            model=self.diffusion,
            condition=condition,
            device=device,
        )

        # Step 2: PINN Refinement
        refined = self.pinn(coarse=coarse, zero_filled=condition)

        # Step 3: Enforce data consistency on final image if requested
        if self.enforce_dc and y_measured is not None and mask is not None:
            refined = apply_data_consistency(
                x_image=refined,
                k_measured=y_measured,
                mask=mask,
                return_complex=False,
            )
            refined = torch.clamp(refined, 0.0, 1.0)

        return coarse, refined

    def forward(
        self,
        condition: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        y_measured: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Unified forward call: calls forward_train if target provided, else sample."""
        if target is not None:
            return self.forward_train(
                target=target,
                condition=condition,
                y_measured=y_measured,
                mask=mask,
            )
        else:
            return self.sample(
                condition=condition,
                y_measured=y_measured,
                mask=mask,
            )

