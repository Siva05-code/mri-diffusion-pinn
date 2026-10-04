"""
Physics-guided conditional diffusion reconstructor.

Training
--------
    residual   = (target - zero_filled) / residual_scale
    x_t        = q_sample(residual, t, noise)
    pred       = UNet([x_t, zf, mask], t)
    x0_res     = to_x0(x_t, t, pred)
    pred_image = clamp(zf + x0_res * residual_scale, 0, 1)

    L = L_diffusion + l1*L_image + l2*L_kspace + l3*L_gradient

Sampling
--------
Deterministic DDIM, with hard Fourier data consistency applied to the clean
image estimate at every reverse step, then the eps estimate recomputed from the
corrected x_0 so the trajectory stays self-consistent.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.losses.composite_loss import LossWeights, composite_loss
from src.models.diffusion_schedule import GaussianDiffusion
from src.models.unet import ConditionalDiffusionUNet
from src.physics.fft import data_consistency, image_to_kspace


class DiffusionReconstructor(nn.Module):
    def __init__(
        self,
        image_size: int = 320,
        diffusion_steps: int = 1000,
        base_channels: int = 48,
        channel_mult=(1, 2, 4, 8),
        time_dim: int = 256,
        attention_levels=(2, 3),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        schedule: str = "cosine",
        prediction_type: str = "v",
        residual_scale: float = 0.1,
        use_mask_conditioning: bool = True,
        min_snr_gamma: float = 5.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.use_mask_conditioning = use_mask_conditioning
        self.min_snr_gamma = float(min_snr_gamma)

        # residual_scale is state, not a hyper-parameter to be re-derived at
        # load time: it must be identical in training and sampling or the
        # reconstruction is silently rescaled. Stored as a buffer so it travels
        # inside the checkpoint.
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))

        self.diffusion = GaussianDiffusion(
            steps=diffusion_steps,
            schedule=schedule,
            prediction_type=prediction_type,
        )
        self.net = ConditionalDiffusionUNet(
            in_channels=3,
            out_channels=1,
            base=base_channels,
            channel_mult=channel_mult,
            time_dim=time_dim,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    def _mask_channel(self, mask: torch.Tensor) -> torch.Tensor:
        """Ablation switch: feeding a constant kills the mask conditioning."""
        return mask if self.use_mask_conditioning else torch.ones_like(mask)

    def set_residual_scale(self, value: float) -> None:
        self.residual_scale.fill_(float(value))

    @property
    def scale(self) -> torch.Tensor:
        return self.residual_scale

    # ------------------------------------------------------------------
    def training_loss(
        self,
        target: torch.Tensor,
        zf: torch.Tensor,
        residual: torch.Tensor,
        mask: torch.Tensor,
        measured_kspace: torch.Tensor,
        weights: LossWeights,
    ) -> dict[str, torch.Tensor]:
        b = residual.shape[0]
        device = residual.device

        t = torch.randint(0, self.diffusion.steps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(residual)
        x_t = self.diffusion.q_sample(residual, t, noise)

        pred = self.net(x_t, zf, self._mask_channel(mask), t)
        diffusion_target = self.diffusion.target_for(residual, noise, t)

        per_sample = ((pred - diffusion_target) ** 2).flatten(1).mean(dim=1)
        if self.min_snr_gamma > 0:
            w = self.diffusion.min_snr_weight(t, self.min_snr_gamma)
            # Renormalise to mean 1 so the weighting rebalances timesteps without
            # silently shrinking the diffusion term relative to the image losses.
            w = w / w.mean().clamp_min(1e-8)
            per_sample = per_sample * w
        loss_diffusion = per_sample.mean()

        x0_res = self.diffusion.to_x0(x_t, t, pred).clamp(-1.0, 1.0)
        pred_image = (zf + x0_res * self.residual_scale).clamp(0.0, 1.0)

        return composite_loss(
            loss_diffusion,
            pred_image,
            target,
            measured_kspace,
            mask,
            weights,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _single_pass(
        self,
        zf,
        mask,
        measured_kspace,
        inference_steps,
        eta,
        use_data_consistency,
        dc_lambda,
        t_start_frac,
        generator,
    ):
        device = zf.device
        mask_in = self._mask_channel(mask)
        steps = self.diffusion.steps

        # --- warm start ---------------------------------------------------
        # Starting the reverse chain at pure noise (t = T) throws away the
        # zero-filled image and makes the network re-derive it from scratch.
        # We already know the answer to within a small residual, so start part
        # way down the chain instead: x_t = q_sample(0, t_start, noise), which
        # is the correctly-noised version of "the residual is zero".
        #
        # How much this helps is model- and data-dependent, and it is NOT
        # reliably the biggest lever: on a short phantom run a pure-noise start
        # scored as well as any warm start. Sweep it with
        # scripts/tune_inference.py rather than trusting a default.
        t_start = int(round(max(0.0, min(1.0, t_start_frac)) * (steps - 1)))
        t_start = max(t_start, 1)

        noise = torch.randn(zf.shape, device=device, dtype=zf.dtype, generator=generator)
        t0 = torch.full((zf.shape[0],), t_start, device=device, dtype=torch.long)
        x = self.diffusion.q_sample(torch.zeros_like(zf), t0, noise)

        timesteps = torch.linspace(
            t_start, 0, max(2, min(inference_steps, t_start + 1)), device=device
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)

        pred_image = zf
        for i, t_val in enumerate(timesteps):
            t = torch.full((zf.shape[0],), int(t_val), device=device, dtype=torch.long)

            pred = self.net(x, zf, mask_in, t)
            x0_res = self.diffusion.to_x0(x, t, pred).clamp(-1.0, 1.0)
            pred_image = (zf + x0_res * self.residual_scale).clamp(0.0, 1.0)

            if use_data_consistency:
                pred_image = data_consistency(
                    pred_image, measured_kspace, mask, lam=dc_lambda
                )

            if i == len(timesteps) - 1:
                break

            x0_res = ((pred_image - zf) / self.residual_scale).clamp(-1.0, 1.0)
            eps = self.diffusion.to_eps(x, t, x0_res)
            t_prev = torch.full(
                (zf.shape[0],), int(timesteps[i + 1]), device=device, dtype=torch.long
            )
            x = self.diffusion.ddim_step(x0_res, eps, t_prev, eta=eta, t=t)

        return pred_image

    @torch.no_grad()
    def reconstruct(
        self,
        zf: torch.Tensor,
        mask: torch.Tensor,
        measured_kspace: torch.Tensor,
        inference_steps: int = 50,
        eta: float = 0.0,
        use_data_consistency: bool = True,
        dc_lambda: float = 1.0,
        generator: torch.Generator | None = None,
        t_start_frac: float = 0.6,
        num_samples: int = 1,
        tta: bool = False,
    ) -> torch.Tensor:
        """
        Reverse diffusion with Fourier data consistency in the loop.

        t_start_frac : where on the noise schedule to begin. 1.0 is the textbook
                       "start from pure noise"; lower values keep more of the
                       zero-filled image. The useful range is usually 0.2-1.0
                       and the optimum depends on how long the model trained;
                       measure it, do not assume it.
        num_samples  : average this many independent samples. PSNR rewards the
                       posterior mean, and a single diffusion draw is a sample
                       from the posterior, not its mean. In every sweep run so
                       far this has been the most dependable single gain, at
                       linear cost in inference time.
        tta          : also reconstruct horizontally- and vertically-flipped
                       copies and average them back. Costs 4x inference and has
                       been observed to *hurt* when the anatomy is not
                       flip-symmetric, so verify it before enabling it.

        ``use_data_consistency=False`` is the ablation path.
        """
        variants = [(False, False)]
        if tta:
            variants += [(True, False), (False, True), (True, True)]

        accum = torch.zeros_like(zf)
        count = 0

        for flip_w, flip_h in variants:
            dims = []
            if flip_w:
                dims.append(-1)
            if flip_h:
                dims.append(-2)

            if dims:
                zf_v = torch.flip(zf, dims)
                mask_v = torch.flip(mask, dims)
                # The flipped zero-filled image already has its spectrum confined
                # to the flipped mask, so this is the exact flipped measurement
                # rather than an approximation of it.
                meas_v = image_to_kspace(zf_v.float()) * mask_v.float()
            else:
                zf_v, mask_v, meas_v = zf, mask, measured_kspace

            for _ in range(max(1, num_samples)):
                out = self._single_pass(
                    zf_v, mask_v, meas_v, inference_steps, eta,
                    use_data_consistency, dc_lambda, t_start_frac, generator,
                )
                accum = accum + (torch.flip(out, dims) if dims else out)
                count += 1

        averaged = accum / count

        # Averaging is a linear operation on data-consistent images, so the mean
        # is still data-consistent; the final projection just removes the
        # accumulated float error.
        if use_data_consistency:
            averaged = data_consistency(averaged, measured_kspace, mask, lam=dc_lambda)
        return averaged


class EMA:
    """
    Exponential moving average of the model weights.

    Buffers are copied rather than averaged: the diffusion schedule and the
    residual scale are constants, and averaging them would drift the sampler
    away from the schedule it was trained on.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state):
        self.shadow.load_state_dict(state)
