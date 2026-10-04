"""
Reconstruction metrics.

All metrics are computed per slice and then averaged, which is what the fastMRI
leaderboard does. Averaging over a flattened batch instead lets bright slices
dominate and makes numbers from different batch sizes incomparable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB, averaged over the batch."""
    pred = pred.detach().float()
    target = target.detach().float()
    dims = tuple(range(1, pred.ndim))
    mse = ((pred - target) ** 2).mean(dim=dims).clamp_min(1e-12)
    return float((10.0 * torch.log10(data_range ** 2 / mse)).mean())


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    """Normalised mean squared error, averaged over the batch."""
    pred = pred.detach().float()
    target = target.detach().float()
    dims = tuple(range(1, pred.ndim))
    num = ((pred - target) ** 2).sum(dim=dims)
    den = (target ** 2).sum(dim=dims).clamp_min(eps)
    return float((num / den).mean())


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Gaussian-window SSIM implemented in torch.

    Kept dependency-free and differentiable so it runs on GPU during validation
    without a round trip to numpy, and so the project does not break when
    scikit-image changes its default window.
    """
    pred = pred.detach().float()
    target = target.detach().float()
    c = pred.shape[1]

    window = _gaussian_window(win_size, sigma, pred.device, pred.dtype)
    window = window.expand(c, 1, win_size, win_size).contiguous()
    pad = win_size // 2

    def filt(x):
        return F.conv2d(F.pad(x, [pad] * 4, mode="reflect"), window, groups=c)

    mu1, mu2 = filt(pred), filt(target)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1 = filt(pred * pred) - mu1_sq
    sigma2 = filt(target * target) - mu2_sq
    sigma12 = filt(pred * target) - mu12

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    return float(ssim_map.flatten(1).mean(dim=1).mean())


def summarize(
    pred: torch.Tensor,
    target: torch.Tensor,
    dc: float | torch.Tensor,
    data_range: float = 1.0,
) -> dict[str, float]:
    return {
        "psnr": psnr(pred, target, data_range),
        "ssim": ssim(pred, target, data_range),
        "nmse": nmse(pred, target),
        "dc_error": float(dc),
    }


class MetricTracker:
    """Running mean of a set of metrics."""

    def __init__(self):
        self.store: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.store.setdefault(k, []).append(float(v))

    def mean(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.store.items()}

    def std(self) -> dict[str, float]:
        return {k: float(np.std(v)) for k, v in self.store.items()}

    def __len__(self) -> int:
        return len(next(iter(self.store.values()))) if self.store else 0
