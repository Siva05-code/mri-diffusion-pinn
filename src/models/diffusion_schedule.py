"""
Diffusion Noise Schedule & DDPM Sampling Utilities.

Implements standard linear beta scheduling, forward Gaussian diffusion (noise injection),
differentiable Tweedie clean image estimation (x0 prediction), and DDPM reverse diffusion sampling.
"""

from typing import Optional
import torch

from src.physics.fft import image_to_kspace_torch, kspace_to_image_torch


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
        mask: Optional[torch.Tensor] = None,
        y_raw: Optional[torch.Tensor] = None,
        mask_raw: Optional[torch.Tensor] = None,
        phase_zf: Optional[torch.Tensor] = None,
        k_measured: Optional[torch.Tensor] = None,
        measurement_guided: bool = True,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Measurement-guided reverse diffusion sampling for accelerated MRI reconstruction:

        At each reverse diffusion timestep t (from T-1 down to 0):
        1. Conditional diffusion model predicts injected noise:
               epsilon_hat = model(x_t, condition, mask, t)
        2. Analytical clean image estimation via Tweedie's formula:
               x0_pred = clamp((x_t - sqrt(1 - alpha_bar_t) * epsilon_hat) / sqrt(alpha_bar_t), 0, 1)
        3. Measurement consistency:
           a) Raw acquisition domain consistency (preferred):
              Uses raw acquired measurements y_raw and raw mask mask_raw via MRIForwardOperator.data_consistency:
                  k_pred = F_raw(Z(x0_pred * exp(i * phase_zf)))
                  k_guided = mask_raw * y_raw + (1 - mask_raw) * k_pred
                  x0_guided = clamp(Re(C(IFFT_raw(k_guided))), 0, 1)
           b) Reconstruction-domain Fourier consistency (fallback when raw k-space is not supplied):
                  k_pred = FFT(x0_pred)
                  k_ref = k_measured if k_measured is not None else FFT(condition)
                  k_guided = M * k_ref + (1 - M) * k_pred
                  x0_guided = clamp(Re(IFFT(k_guided)), 0, 1)
        4. Measurement-guided DDPM reverse update:
           Computes the reverse posterior mean mu_t guided by x0_guided:
               mu_t = (sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)) * x0_guided
                    + (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x_t
               x_{t-1} = mu_t + sqrt(posterior_variance) * z  (z ~ N(0, I) for t > 0, 0 for t = 0)

        Args:
            model: Conditional diffusion U-Net (expects (x_t, condition, mask, t)).
            condition: Zero-filled initial MRI reconstruction condition [B, 1, H, W].
            mask: 2D/4D spatial conditioning mask [B, 1, H, W] for the neural network.
            y_raw: Optional acquired raw scanner k-space measurements [B, H_raw, W_raw].
            mask_raw: Optional binary acquisition mask [B, H_raw, W_raw] or [1, W_raw].
            phase_zf: Optional spatial phase from complex zero-filled reconstruction [B, 1, H, W].
            k_measured: Optional fallback reconstruction-domain Fourier reference.
            measurement_guided: Whether to enforce measurement consistency.
            device: Execution device (defaults to condition.device).

        Returns:
            torch.Tensor: Reconstructed clean image, clamped to [0, 1].
        """
        from src.physics.forward_operator import MRIForwardOperator
        forward_op = MRIForwardOperator()

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
                mask,
                t,
            )

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            alpha_bar_prev = self.alpha_bars[i - 1] if i > 0 else torch.tensor(1.0, device=device)
            beta = self.betas[i]

            # 1. Tweedie clean image prediction
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
            pred_x0 = (x - sqrt_one_minus_alpha_bar * predicted_noise) / (sqrt_alpha_bar + 1e-8)
            pred_x0 = torch.clamp(pred_x0, 0.0, 1.0)

            # 2. Measurement guidance
            if measurement_guided:
                if y_raw is not None and mask_raw is not None:
                    # Primary: Raw acquisition domain data consistency
                    x0_dc = forward_op.data_consistency(
                        x_crop=pred_x0.squeeze(1),
                        y_raw=y_raw.to(device),
                        mask_raw=mask_raw.to(device),
                        phase=phase_zf.squeeze(1) if phase_zf is not None else None,
                        return_complex=False,
                    )
                    x0_guided = x0_dc.unsqueeze(1).clamp(0.0, 1.0)
                elif mask is not None:
                    # Secondary fallback: common reconstruction-domain Fourier consistency
                    img_complex = pred_x0.squeeze(1).to(torch.complex64)
                    k_pred = image_to_kspace_torch(img_complex)

                    if k_measured is not None:
                        k_ref = k_measured.to(device)
                    else:
                        cond_complex = condition.squeeze(1).to(torch.complex64)
                        k_ref = image_to_kspace_torch(cond_complex)

                    mask_k = mask
                    while mask_k.ndim > 3:
                        mask_k = mask_k.squeeze(1)
                    if mask_k.shape[-1] > k_pred.shape[-1]:
                        w_from = (mask_k.shape[-1] - k_pred.shape[-1]) // 2
                        mask_k = mask_k[..., w_from : w_from + k_pred.shape[-1]]
                    if mask_k.ndim > 2 and mask_k.shape[-2] > k_pred.shape[-2]:
                        h_from = (mask_k.shape[-2] - k_pred.shape[-2]) // 2
                        mask_k = mask_k[..., h_from : h_from + k_pred.shape[-2], :]
                    mask_k = mask_k.to(device=device, dtype=k_pred.real.dtype)

                    k_guided = mask_k * k_ref + (1.0 - mask_k) * k_pred
                    x_rec = kspace_to_image_torch(k_guided)
                    x0_guided = torch.real(x_rec).unsqueeze(1).clamp(0.0, 1.0)
                else:
                    x0_guided = pred_x0
            else:
                x0_guided = pred_x0

            if i > 0:
                noise = torch.randn_like(x, device=device)
            else:
                noise = torch.zeros_like(x, device=device)

            # 3. Measurement-guided DDPM reverse update step
            coeff_x0 = (torch.sqrt(alpha_bar_prev) * beta) / (1.0 - alpha_bar)
            coeff_xt = (torch.sqrt(alpha) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar)

            # Correct DDPM posterior variance:
            # beta_tilde = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
            posterior_variance = (
                beta * (1.0 - alpha_bar_prev)
            ) / (1.0 - alpha_bar)

            posterior_variance = torch.clamp(
                posterior_variance,
                min=1e-20,
            )

            x = (
                coeff_x0 * x0_guided
                + coeff_xt * x
                + torch.sqrt(posterior_variance) * noise
            )

        return x0_guided.clamp(0.0, 1.0)