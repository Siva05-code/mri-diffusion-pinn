"""
MRI Data Preprocessing Utilities.

Provides functions to compute magnitude images, normalize k-space and reconstructed images,
center crop images, and format complex numbers for PyTorch processing.
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


def normalize_zero_one(data: Union[np.ndarray, torch.Tensor], eps: float = 1e-12):
    """Normalize array/tensor to [0, 1] range."""
    d_min = data.min()
    d_max = data.max()
    return (data - d_min) / (d_max - d_min + eps)


def normalize_instance(data: torch.Tensor, eps: float = 1e-11):
    """Normalize data by subtracting mean and dividing by standard deviation."""
    mean = data.mean()
    std = data.std()
    return (data - mean) / (std + eps), mean, std


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
        # If input is smaller than requested target, preserve native dimensions
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
    Ensures zero-filled input and reference target undergo consistent spatial preprocessing:
    1. If target_image is provided:
       - If target is pre-cropped (e.g., fastMRI reconstruction_rss at 320x320), center-crop zf_image to match.
       - If both are raw IFFT (640x368) and crop_size is given, center-crop both to crop_size.
       - Guarantees identical (320, 320) spatial dimensions without bilinear/bicubic distortion.
       - Normalizes both inputs to [0, 1] consistently and returns (zf_norm, target_norm).
    2. If target_image is None:
       - Center-crops zf_image to crop_size (if specified) and normalizes to [0, 1].
       - Returns zf_norm.

    Args:
        zf_image: Zero-filled magnitude image tensor [H_zf, W_zf]
        target_image: Optional reference target magnitude image tensor [H_tgt, W_tgt]
        crop_size: Standard evaluation FOV (default: (320, 320))

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
            (zf_norm, target_norm) if target_image is provided, else zf_norm.
    """
    if target_image is None:
        if crop_size is not None and zf_image.shape[-2:] != crop_size:
            if zf_image.shape[-2] >= crop_size[0] and zf_image.shape[-1] >= crop_size[1]:
                zf_image = center_crop(zf_image, crop_size)
        return normalize_zero_one(zf_image)

    # Both images provided:
    # Case 1: Target is already pre-cropped (e.g., fastMRI reconstruction_rss at 320x320)
    if zf_image.shape != target_image.shape:
        target_shape = (target_image.shape[-2], target_image.shape[-1])
        if zf_image.shape[-2] >= target_shape[0] and zf_image.shape[-1] >= target_shape[1]:
            zf_image = center_crop(zf_image, target_shape)
        elif target_image.shape[-2] >= zf_image.shape[-2] and target_image.shape[-1] >= zf_image.shape[-1]:
            target_image = center_crop(target_image, (zf_image.shape[-2], zf_image.shape[-1]))

    # Case 2: Both are still full k-space IFFT (640x368); crop both to standard FOV if requested
    if crop_size is not None and zf_image.shape[-2:] != crop_size:
        if zf_image.shape[-2] >= crop_size[0] and zf_image.shape[-1] >= crop_size[1]:
            zf_image = center_crop(zf_image, crop_size)
            target_image = center_crop(target_image, crop_size)

    assert zf_image.shape == target_image.shape, (
        f"Spatial dimension mismatch after alignment: zf {zf_image.shape} vs target {target_image.shape}"
    )

    zf_norm = normalize_zero_one(zf_image)
    target_norm = normalize_zero_one(target_image)
    return zf_norm, target_norm
