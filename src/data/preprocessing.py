"""
Slice-level preprocessing for fastMRI single-coil data.

Pipeline for one slice:

    raw k-space (640 x 372, complex)
            |
            v   centered IFFT
    complex image
            |
            v   centre crop to 320 x 320
    cropped complex image
            |
            v   magnitude + intensity normalisation
    target in [0, 1]   <-- this is the ground truth the network sees
"""

from __future__ import annotations

import numpy as np


def center_crop(x: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop the last two axes to (size, size), zero-padding if needed."""
    h, w = x.shape[-2:]

    if h < size or w < size:
        pad_h = max(size - h, 0)
        pad_w = max(size - w, 0)
        pad = [(0, 0)] * (x.ndim - 2) + [
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        ]
        x = np.pad(x, pad)
        h, w = x.shape[-2:]

    top = (h - size) // 2
    left = (w - size) // 2
    return x[..., top:top + size, left:left + size]


def normalize_magnitude(
    x: np.ndarray,
    percentile: float = 99.9,
) -> tuple[np.ndarray, float]:
    """
    Scale a magnitude image into [0, 1].

    A high percentile rather than the raw maximum is used so that a handful of
    hot pixels cannot compress the whole image into a narrow band. The scale is
    returned so metrics can be mapped back to the original intensity range if
    needed.
    """
    mag = np.abs(x).astype(np.float32)
    scale = float(np.percentile(mag, percentile))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(mag.max()) if mag.max() > 0 else 1.0
    out = np.clip(mag / scale, 0.0, 1.0)
    return out.astype(np.float32), scale


def kspace_to_target(
    kspace: np.ndarray,
    image_size: int = 320,
    percentile: float = 99.9,
) -> tuple[np.ndarray, float]:
    """Full raw-k-space -> normalised 320x320 magnitude target."""
    k = np.asarray(kspace)
    if k.ndim != 2:
        raise ValueError(
            f"Expected 2-D single-coil k-space for one slice, got shape {k.shape}. "
            "This project supports the fastMRI *single-coil* knee dataset."
        )

    # Centered IFFT, done in numpy so the dataset stays torch-free in workers.
    img = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k), norm="ortho")
    )
    img = center_crop(img, image_size)
    return normalize_magnitude(img, percentile)
