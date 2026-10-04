"""
Gaussian diffusion schedule.

Supports both eps-prediction and v-prediction. v-prediction is the default
because of how this project uses the model: the training loss and the sampler
both need a clean-signal estimate x_0 at *every* timestep, including t close to
T. With eps-prediction that estimate is

    x_0 = (x_t - sqrt(1 - abar) * eps) / sqrt(abar)

and sqrt(abar) falls to ~1e-3 at the end of the schedule, so a small error in
eps is amplified by a factor of a thousand. The v-parameterisation

    v   = sqrt(abar) * eps - sqrt(1 - abar) * x_0
    x_0 = sqrt(abar) * x_t - sqrt(1 - abar) * v

involves no division and stays well conditioned everywhere.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def linear_beta_schedule(steps: int, beta_start=1e-4, beta_end=0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule; keeps more signal at large t."""
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    abar = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    abar = abar / abar[0]
    betas = 1.0 - (abar[1:] / abar[:-1])
    return betas.clamp(1e-8, 0.999)


def extract(arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Gather per-sample schedule values and reshape for broadcasting."""
    out = arr.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (ndim - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        steps: int = 1000,
        schedule: str = "cosine",
        prediction_type: str = "v",
    ):
        super().__init__()
        if prediction_type not in ("v", "eps"):
            raise ValueError("prediction_type must be 'v' or 'eps'")

        self.steps = int(steps)
        self.prediction_type = prediction_type

        betas = (
            cosine_beta_schedule(self.steps)
            if schedule == "cosine"
            else linear_beta_schedule(self.steps)
        )
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("abar", abar.float())
        self.register_buffer("sqrt_abar", abar.sqrt().float())
        self.register_buffer("sqrt_one_minus_abar", (1.0 - abar).sqrt().float())

    # ------------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = extract(self.sqrt_abar, t, x0.ndim)
        b = extract(self.sqrt_one_minus_abar, t, x0.ndim)
        return a * x0 + b * noise

    def target_for(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The quantity the network is trained to regress."""
        if self.prediction_type == "eps":
            return noise
        a = extract(self.sqrt_abar, t, x0.ndim)
        b = extract(self.sqrt_one_minus_abar, t, x0.ndim)
        return a * noise - b * x0

    def to_x0(self, xt: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        a = extract(self.sqrt_abar, t, xt.ndim)
        b = extract(self.sqrt_one_minus_abar, t, xt.ndim)
        if self.prediction_type == "eps":
            return (xt - b * pred) / a.clamp_min(1e-4)
        return a * xt - b * pred

    def to_eps(self, xt: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Recover eps consistent with a (possibly corrected) x_0 estimate."""
        a = extract(self.sqrt_abar, t, xt.ndim)
        b = extract(self.sqrt_one_minus_abar, t, xt.ndim)
        return (xt - a * x0) / b.clamp_min(1e-4)

    # ------------------------------------------------------------------
    def ddim_step(
        self,
        x0: torch.Tensor,
        eps: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        One DDIM transition from t to t_prev given x_0 and eps estimates.

        eta = 0 is deterministic DDIM; eta = 1 reproduces the DDPM posterior.
        """
        abar_prev = extract(self.abar, t_prev, x0.ndim)

        sigma = torch.zeros_like(abar_prev)
        if eta > 0.0 and t is not None:
            abar_t = extract(self.abar, t, x0.ndim)
            sigma = eta * torch.sqrt(
                ((1 - abar_prev) / (1 - abar_t).clamp_min(1e-8))
                * (1 - abar_t / abar_prev.clamp_min(1e-8)).clamp_min(0.0)
            )

        dir_xt = torch.sqrt((1.0 - abar_prev - sigma ** 2).clamp_min(0.0)) * eps
        x_prev = torch.sqrt(abar_prev) * x0 + dir_xt
        if eta > 0.0:
            x_prev = x_prev + sigma * torch.randn_like(x_prev)
        return x_prev

    def min_snr_weight(self, t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
        """
        Min-SNR-gamma loss weighting (Hang et al., 2023).

        Without it the per-timestep losses are wildly unbalanced: low-t terms
        have enormous SNR and dominate the gradient, so the model spends its
        capacity on timesteps that were already easy. Clipping the effective
        weight at gamma rebalances them and is one of the cheapest reliable
        quality gains available -- typically a few tenths of a dB for one line
        of code.
        """
        abar = self.abar.gather(0, t)
        snr = abar / (1.0 - abar).clamp_min(1e-8)
        clipped = torch.clamp(snr, max=gamma)
        if self.prediction_type == "v":
            return clipped / (snr + 1.0)
        return clipped / snr.clamp_min(1e-8)

    def timestep_sequence(self, inference_steps: int, device) -> torch.Tensor:
        """Descending timestep grid used at inference, always ending at t = 0."""
        inference_steps = max(2, min(inference_steps, self.steps))
        seq = torch.linspace(
            self.steps - 1, 0, inference_steps, device=device
        ).round().long()
        return torch.unique_consecutive(seq)
