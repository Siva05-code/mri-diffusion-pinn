"""
Diffusion Noise Schedule & DDPM Sampling Utilities.

Implements:
- Linear beta scheduling
- Forward Gaussian diffusion
- Tweedie clean-image estimation
- Measurement-guided DDPM reverse diffusion
- Fourier-domain data consistency
"""

from typing import Optional

import torch

from src.physics.fft import (
    image_to_kspace_torch,
    kspace_to_image_torch,
)


class DiffusionSchedule:
    """
    Linear beta schedule for conditional Gaussian diffusion models.

    Args:
        timesteps (int): Number of diffusion timesteps.
        beta_start (float): Starting beta value.
        beta_end (float): Ending beta value.
        device (str or torch.device): Device for schedule tensors.
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
        """Move schedule tensors to the specified device."""

        self.device = torch.device(device)

        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_bars = self.alpha_bars.to(self.device)

        return self

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion process.

        q(x_t | x_0)
        = N(
            sqrt(alpha_bar_t) * x_0,
            (1 - alpha_bar_t) * I
        )

        x_t =
            sqrt(alpha_bar_t) * x_0
            + sqrt(1 - alpha_bar_t) * epsilon
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
        Estimate clean image x0 using Tweedie's formula.
        """

        alpha_bar = self.alpha_bars[t].to(x_t.device)
        alpha_bar = alpha_bar[:, None, None, None]

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(
            1.0 - alpha_bar
        )

        pred_x0 = (
            x_t
            - sqrt_one_minus_alpha_bar * predicted_noise
        ) / (sqrt_alpha_bar + 1e-8)

        if clamp:
            pred_x0 = torch.clamp(
                pred_x0,
                0.0,
                1.0,
            )

        return pred_x0

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        k_measured: Optional[torch.Tensor] = None,
        measurement_guided: bool = True,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Measurement-guided DDPM reverse diffusion.

        At every reverse timestep:

        1. Predict noise using the conditional diffusion model.
        2. Estimate x0 using Tweedie's formula.
        3. Enforce Fourier-domain measurement consistency.
        4. Compute the DDPM posterior mean.
        5. Add the correct DDPM posterior variance noise.

        The Fourier consistency operates in the common
        reconstruction domain.

        Args:
            model:
                Conditional diffusion U-Net.

            condition:
                Zero-filled MRI reconstruction.

            mask:
                Binary sampling mask.

            k_measured:
                Optional measured k-space in the same
                reconstruction domain.

            measurement_guided:
                Whether to enforce Fourier consistency.

            device:
                Sampling device.

        Returns:
            Reconstructed image in [0, 1].
        """

        if device is None:
            device = condition.device
        else:
            device = torch.device(device)

        model.eval()
        self.to(device)

        # --------------------------------------------------
        # INITIALIZATION
        # --------------------------------------------------
        # Start reverse diffusion from Gaussian noise.
        x = torch.randn_like(
            condition,
            device=device,
        )

        # --------------------------------------------------
        # REVERSE DIFFUSION
        # --------------------------------------------------
        for i in reversed(range(self.timesteps)):

            t = torch.full(
                (condition.shape[0],),
                i,
                device=device,
                dtype=torch.long,
            )

            # --------------------------------------------------
            # 1. Predict noise
            # --------------------------------------------------
            predicted_noise = model(
                x,
                condition,
                mask,
                t,
            )

            # --------------------------------------------------
            # Schedule values
            # --------------------------------------------------
            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]

            if i > 0:
                alpha_bar_prev = self.alpha_bars[i - 1]
            else:
                alpha_bar_prev = torch.tensor(
                    1.0,
                    device=device,
                    dtype=alpha_bar.dtype,
                )

            beta = self.betas[i]

            # --------------------------------------------------
            # 2. Tweedie x0 prediction
            # --------------------------------------------------
            sqrt_alpha_bar = torch.sqrt(alpha_bar)

            sqrt_one_minus_alpha_bar = torch.sqrt(
                1.0 - alpha_bar
            )

            pred_x0 = (
                x
                - sqrt_one_minus_alpha_bar * predicted_noise
            ) / (sqrt_alpha_bar + 1e-8)

            pred_x0 = torch.clamp(
                pred_x0,
                0.0,
                1.0,
            )

            # --------------------------------------------------
            # 3. Fourier measurement consistency
            # --------------------------------------------------
            if measurement_guided and mask is not None:

                # Convert predicted magnitude image
                # to complex representation.
                img_complex = (
                    pred_x0
                    .squeeze(1)
                    .to(torch.complex64)
                )

                k_pred = image_to_kspace_torch(
                    img_complex
                )

                # --------------------------------------------------
                # Measurement reference
                # --------------------------------------------------
                if k_measured is not None:

                    k_ref = k_measured.to(
                        device=device,
                        dtype=k_pred.dtype,
                    )

                else:

                    # Use Fourier representation of the
                    # zero-filled conditioning image.
                    cond_complex = (
                        condition
                        .squeeze(1)
                        .to(torch.complex64)
                    )

                    k_ref = image_to_kspace_torch(
                        cond_complex
                    )

                # --------------------------------------------------
                # Prepare mask
                # --------------------------------------------------
                mask_k = mask

                while mask_k.ndim > 3:
                    mask_k = mask_k.squeeze(1)

                # Center-crop width if necessary.
                if mask_k.shape[-1] > k_pred.shape[-1]:

                    w_from = (
                        mask_k.shape[-1]
                        - k_pred.shape[-1]
                    ) // 2

                    mask_k = mask_k[
                        ...,
                        w_from:w_from + k_pred.shape[-1],
                    ]

                # Center-crop height if necessary.
                if (
                    mask_k.ndim > 2
                    and mask_k.shape[-2] > k_pred.shape[-2]
                ):

                    h_from = (
                        mask_k.shape[-2]
                        - k_pred.shape[-2]
                    ) // 2

                    mask_k = mask_k[
                        ...,
                        h_from:h_from + k_pred.shape[-2],
                        :,
                    ]

                mask_k = mask_k.to(
                    device=device,
                    dtype=k_pred.real.dtype,
                )

                # --------------------------------------------------
                # Hard Fourier data consistency
                # --------------------------------------------------
                k_guided = (
                    mask_k * k_ref
                    + (1.0 - mask_k) * k_pred
                )

                # Back to image domain.
                x_rec = kspace_to_image_torch(
                    k_guided
                )

                x0_guided = (
                    torch.real(x_rec)
                    .unsqueeze(1)
                )

                x0_guided = torch.clamp(
                    x0_guided,
                    0.0,
                    1.0,
                )

            else:

                x0_guided = pred_x0

            # --------------------------------------------------
            # 4. DDPM posterior mean
            # --------------------------------------------------
            coeff_x0 = (
                torch.sqrt(alpha_bar_prev)
                * beta
            ) / (
                1.0 - alpha_bar
            )

            coeff_xt = (
                torch.sqrt(alpha)
                * (
                    1.0 - alpha_bar_prev
                )
            ) / (
                1.0 - alpha_bar
            )

            # --------------------------------------------------
            # 5. CORRECT DDPM POSTERIOR VARIANCE
            # --------------------------------------------------
            #
            # beta_tilde =
            # beta_t * (1-alpha_bar_{t-1})
            # ----------------------------
            #      (1-alpha_bar_t)
            #
            # This is the posterior variance
            # corresponding to the posterior mean above.
            #
            posterior_variance = (
                beta
                * (
                    1.0 - alpha_bar_prev
                )
            ) / (
                1.0 - alpha_bar
            )

            posterior_variance = torch.clamp(
                posterior_variance,
                min=1e-20,
            )

            # --------------------------------------------------
            # 6. Sample x_{t-1}
            # --------------------------------------------------
            if i > 0:

                noise = torch.randn_like(
                    x,
                    device=device,
                )

            else:

                # No noise at final timestep.
                noise = torch.zeros_like(
                    x,
                    device=device,
                )

            x = (
                coeff_x0 * x0_guided
                + coeff_xt * x
                + torch.sqrt(
                    posterior_variance
                ) * noise
            )

        # --------------------------------------------------
        # FINAL OUTPUT
        # --------------------------------------------------
        return torch.clamp(
            x0_guided,
            0.0,
            1.0,
        )