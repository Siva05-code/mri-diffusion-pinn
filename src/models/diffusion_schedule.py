"""
Diffusion Noise Schedule & DDPM Sampling Utilities.

Implements standard linear beta scheduling, forward Gaussian diffusion (noise injection),
differentiable Tweedie clean image estimation (x0 prediction), and DDPM reverse diffusion sampling.
"""

from typing import Optional
import torch


class DiffusionSchedule:
    """
    Linear beta schedule for conditional Gaussian diffusion models.

    Args:
        timesteps (int): Number of diffusion timesteps (default: 100).
        beta_start (float): Starting variance schedule (default: 1e-4).
        beta_end (float): Ending variance schedule (default: 0.02).
        device (str or torch.device): Device on which schedule tensors reside.
    """
    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.timesteps = timesteps
        self.device = torch.device(device)

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            device=self.device,
            dtype=torch.float32,
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0,
        )

    def to(self, device: torch.device):
        """Moves all schedule tensors to specified device."""
        self.device = torch.device(device)
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_bars = self.alpha_bars.to(self.device)
        return self

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion process: adds Gaussian noise to clean image x0 at timestep t.
            q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        alpha_bar = self.alpha_bars[t].to(x0.device)
        alpha_bar = alpha_bar[:, None, None, None]

        return (
            torch.sqrt(alpha_bar) * x0
            + torch.sqrt(1.0 - alpha_bar) * noise
        )

    def predict_x0(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
        clamp: bool = True,
    ) -> torch.Tensor:
        """
        Differentiable analytical clean image estimation via Tweedie's formula:
            x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_hat) / sqrt(alpha_bar_t)

        Args:
            x_t: Noisy image at timestep t [B, C, H, W].
            predicted_noise: Predicted noise from conditional diffusion network [B, C, H, W].
            t: Timestep indices [B].
            clamp: Whether to clamp predicted image to valid normalized MRI intensity range [0, 1].

        Returns:
            torch.Tensor: Differentiable clean image estimate x_0_pred [B, C, H, W].
        """
        alpha_bar = self.alpha_bars[t].to(x_t.device)
        alpha_bar = alpha_bar[:, None, None, None]

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        pred_x0 = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / (sqrt_alpha_bar + 1e-8)
        if clamp:
            pred_x0 = torch.clamp(pred_x0, 0.0, 1.0)
        return pred_x0

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        DDPM reverse diffusion sampling:
            p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), beta_t * I)
        where:
            mu_theta = 1 / sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * predicted_noise)
            x_{t-1} = mu_theta + sqrt(beta_t) * z  (for t > 0)

        Args:
            model: Conditional diffusion U-Net (expects (x_t, condition, t)).
            condition: Zero-filled initial MRI reconstruction condition [B, C, H, W].
            device: Execution device (defaults to condition.device).

        Returns:
            torch.Tensor: Reconstructed clean image, clamped to [0, 1] after sampling.
        """
        if device is None:
            device = condition.device
        else:
            device = torch.device(device)

        model.eval()
        self.to(device)

        # Initialize from pure standard Gaussian noise
        x = torch.randn_like(condition, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full(
                (condition.shape[0],),
                i,
                device=device,
                dtype=torch.long,
            )

            predicted_noise = model(
                x,
                condition,
                t,
            )

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x, device=device)
            else:
                noise = torch.zeros_like(x, device=device)

            # DDPM reverse update step
            x = (
                (1.0 / torch.sqrt(alpha))
                * (
                    x
                    - (beta / torch.sqrt(1.0 - alpha_bar))
                    * predicted_noise
                )
                + torch.sqrt(beta) * noise
            )

        # Clamp output to valid normalized intensity range [0, 1] only after sampling is complete
        return x.clamp(0.0, 1.0)