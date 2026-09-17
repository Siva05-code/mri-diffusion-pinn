"""
MRI Data Preprocessing Utilities.

Provides functions to compute magnitude images, normalize k-space and reconstructed images,
center crop images, and format complex numbers for PyTorch processing.

Implements a unified, leakage-free per-slice normalization protocol using a single shared scale factor
derived from the observed zero-filled reconstruction.
"""

from typing import Tuple, Union, Optional
import torch
import numpy as np


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute magnitude of complex tensor.
    Supports complex PyTorch tensors or 2-channel real representation [..., 2].
    """
    if torch.is_complex(data):
        return torch.abs(data)
    elif data.shape[-1] == 2:
        return torch.sqrt(data[..., 0]**2 + data[..., 1]**2 + 1e-12)
    else:
        return torch.abs(data)


def compute_slice_scale(
    image: torch.Tensor,
    percentile: float = 99.0,
    eps: float = 1e-11,
) -> float:
    """
    Compute a robust intensity scaling factor for a slice based on a high intensity quantile.

    Args:
        image: Real or complex tensor.
        percentile: Quantile percentile (default: 99.0).
        eps: Small epsilon to prevent zero division.

    Returns:
        float: Positive scalar scale factor.
    """
    if torch.is_complex(image):
        mag = torch.abs(image)
    else:
        mag = image.abs()

    q = percentile / 100.0
    scale = torch.quantile(mag.float().flatten(), q).item()
    if scale < eps:
        scale = mag.float().max().item()
    return max(float(scale), eps)


def normalize_slice_pair(
    zf_image: torch.Tensor,
    target_image: Optional[torch.Tensor] = None,
    percentile: float = 99.0,
    eps: float = 1e-11,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    """
    Normalize zero-filled input and optional reference target using ONE shared scale factor.

    To strictly guarantee ZERO ground-truth leakage at inference time, the reference scale
    is calculated solely from the zero-filled input condition (or its 99th percentile).
    Both the zero-filled condition and the ground-truth target are divided by this IDENTICAL scale,
    preserving relative image contrast and physical intensity linearity for quantitative metrics (PSNR/SSIM).

    Args:
        zf_image: Zero-filled magnitude image tensor.
        target_image: Optional target magnitude image tensor.
        percentile: Quantile percentile for scale computation (default: 99.0).
        eps: Numerical stability constant.

    Returns:
        Tuple: (zf_norm, target_norm, scale)
    """
    scale = compute_slice_scale(zf_image, percentile=percentile, eps=eps)
    zf_norm = zf_image / scale
    target_norm = (target_image / scale) if target_image is not None else None
    return zf_norm, target_norm, scale


def center_crop(data: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a physical center crop to the last two dimensions of the input tensor.

    In fastMRI knee data, the raw readout dimension (640) includes 2x frequency-encoding
    oversampling to prevent wrap-around aliasing, and phase encoding (368) includes outer padding.
    Center cropping removes this oversampling and padding to isolate the true 320x320 FOV
    without any geometric resampling or aspect ratio distortion.

    Args:
        data (torch.Tensor): Input tensor [..., H, W]
        target_shape (tuple): Target shape (target_h, target_w)
    Returns:
        torch.Tensor: Center-cropped tensor [..., target_h, target_w]
    """
    target_h, target_w = target_shape[-2], target_shape[-1]
    curr_h, curr_w = data.shape[-2], data.shape[-1]

    if curr_h == target_h and curr_w == target_w:
        return data

    if curr_h < target_h or curr_w < target_w:
        return data

    h_from = (curr_h - target_h) // 2
    w_from = (curr_w - target_w) // 2
    return data[..., h_from : h_from + target_h, w_from : w_from + target_w]


def align_and_normalize_slice(
    zf_image: torch.Tensor,
    target_image: Optional[torch.Tensor] = None,
    crop_size: Optional[Tuple[int, int]] = (320, 320),
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Spatial alignment and unified shared-scale normalization.
    Ensures zero-filled and target slices are on the identical intensity scale and FOV.
    """
    if target_image is None:
        if crop_size is not None and zf_image.shape[-2:] != crop_size:
            if zf_image.shape[-2] >= crop_size[0] and zf_image.shape[-1] >= crop_size[1]:
                zf_image = center_crop(zf_image, crop_size)
        zf_norm, _, _ = normalize_slice_pair(zf_image, target_image=None)
        return zf_norm

    # Spatially align dimensions to crop_size
    if zf_image.shape[-2:] != target_image.shape[-2:]:
        target_shape = (target_image.shape[-2], target_image.shape[-1])
        if zf_image.shape[-2] >= target_shape[0] and zf_image.shape[-1] >= target_shape[1]:
            zf_image = center_crop(zf_image, target_shape)
        elif target_image.shape[-2] >= zf_image.shape[-2] and target_image.shape[-1] >= zf_image.shape[-1]:
            target_image = center_crop(target_image, (zf_image.shape[-2], zf_image.shape[-1]))

    if crop_size is not None and zf_image.shape[-2:] != crop_size:
        if zf_image.shape[-2] >= crop_size[0] and zf_image.shape[-1] >= crop_size[1]:
            zf_image = center_crop(zf_image, crop_size)
            target_image = center_crop(target_image, crop_size)

    assert zf_image.shape == target_image.shape, (
        f"Spatial dimension mismatch after alignment: zf {zf_image.shape} vs target {target_image.shape}"
    )

    zf_norm, target_norm, _ = normalize_slice_pair(zf_image, target_image)
    return zf_norm, target_norm
