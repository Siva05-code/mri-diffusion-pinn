"""
Differentiable SSIM loss.

Separate from src/evaluation/metrics.ssim, which detaches its inputs because it
is a reporting function. If you want SSIM to go up, put SSIM in the objective:
L1 and SSIM disagree about what matters (L1 is pointwise, SSIM is local
structure), and optimising only L1 leaves SSIM on the table.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Mean SSIM, kept in the autograd graph."""
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
    return ssim_map.mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    with torch.autocast(device_type=pred.device.type, enabled=False):
        return 1.0 - ssim_index(pred.float(), target.float(), data_range)
