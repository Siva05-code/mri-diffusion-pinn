"""Weighted combination of the four training losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.losses.image_loss import l1_loss, gradient_loss
from src.losses.kspace_loss import kspace_consistency_loss
from src.losses.ssim_loss import ssim_loss


@dataclass
class LossWeights:
    image: float = 1.0
    kspace: float = 0.1
    gradient: float = 0.01
    ssim: float = 0.0


def composite_loss(
    diffusion_loss: torch.Tensor,
    pred_image: torch.Tensor,
    target: torch.Tensor,
    measured_kspace: torch.Tensor,
    mask: torch.Tensor,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    """
    L_total = L_diffusion
            + lambda_image    * L1(pred, target)
            + lambda_kspace   * || mask * (FFT(pred) - k_measured) ||^2
            + lambda_gradient * gradient L1
            + lambda_ssim     * (1 - SSIM)
    """
    l_img = l1_loss(pred_image, target)
    l_k = kspace_consistency_loss(pred_image, measured_kspace, mask)
    l_grad = gradient_loss(pred_image, target)

    total = (
        diffusion_loss
        + weights.image * l_img
        + weights.kspace * l_k
        + weights.gradient * l_grad
    )

    out = {
        "total": total,
        "diffusion": diffusion_loss.detach(),
        "image": l_img.detach(),
        "kspace": l_k.detach(),
        "gradient": l_grad.detach(),
    }

    if weights.ssim > 0:
        l_ssim = ssim_loss(pred_image, target)
        out["total"] = out["total"] + weights.ssim * l_ssim
        out["ssim"] = l_ssim.detach()

    return out
