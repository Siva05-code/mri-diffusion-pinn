"""Fourier-domain measurement consistency loss."""

from __future__ import annotations

import torch

from src.physics.fft import fft2c


def kspace_consistency_loss(
    pred_image: torch.Tensor,
    measured_kspace: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared error on the *measured* Fourier coefficients only.

    Runs in float32 with autocast disabled: complex FFTs have no reliable
    float16 path and this is the usual cause of an AMP crash or of silent
    NaNs halfway through the first epoch.
    """
    with torch.autocast(device_type=pred_image.device.type, enabled=False):
        pred_k = fft2c(pred_image.float())
        diff = mask.float() * (pred_k - measured_kspace.to(torch.complex64))
        return (diff.real ** 2 + diff.imag ** 2).mean()
