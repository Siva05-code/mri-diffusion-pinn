"""
MRI Data Preprocessing Utilities.

Provides functions to compute magnitude images, normalize k-space and reconstructed images,
and format complex numbers for PyTorch processing.
center crop images, and format complex numbers for PyTorch processing.
"""

from typing import Tuple, Union
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


def normalize_zero_one(data: np.ndarray or torch.Tensor, eps: float = 1e-12):
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
    Apply a center crop to the last two dimensions of the input tensor.

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
        # If input is smaller than requested target, no cropping is possible without padding
        return data

    h_from = (curr_h - target_h) // 2
    w_from = (curr_w - target_w) // 2
    return data[..., h_from : h_from + target_h, w_from : w_from + target_w]


def align_and_normalize_slice(
    zf_image: torch.Tensor,
    target_image: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Verifies actual spatial dimensions between zero-filled input and target image,
    applies center cropping if target is cropped (e.g. 320x320 target vs 640x368 input),
    and normalizes both to [0, 1] consistently.

    Args:
        zf_image: Zero-filled magnitude image tensor [H_zf, W_zf]
        target_image: Reference target magnitude image tensor [H_tgt, W_tgt]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (zf_norm, target_norm) matching in spatial shape.
    """
    if zf_image.shape != target_image.shape:
        # Check if target is a center crop of zero-filled image
        if zf_image.shape[-2] >= target_image.shape[-2] and zf_image.shape[-1] >= target_image.shape[-1]:
            zf_image = center_crop(zf_image, (target_image.shape[-2], target_image.shape[-1]))
        elif target_image.shape[-2] >= zf_image.shape[-2] and target_image.shape[-1] >= zf_image.shape[-1]:
            target_image = center_crop(target_image, (zf_image.shape[-2], zf_image.shape[-1]))

    zf_norm = normalize_zero_one(zf_image)
    target_norm = normalize_zero_one(target_image)
    return zf_norm, target_norm
