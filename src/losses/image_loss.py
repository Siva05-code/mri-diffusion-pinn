"""Image-domain losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Smooth L1 variant; less twitchy than pure L1 near zero."""
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """First-order finite-difference loss; encourages edges to survive."""
    px = pred[..., :, 1:] - pred[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)
