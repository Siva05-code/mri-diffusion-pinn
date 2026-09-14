"""
MRI Data Preprocessing Utilities.

Provides functions to compute magnitude images, normalize k-space and reconstructed images,
and format complex numbers for PyTorch processing.
"""

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
    """Normalize array/tensor to [0, 1] range."""
    d_min = data.min()
    d_max = data.max()
    return (data - d_min) / (d_max - d_min + eps)


def normalize_instance(data: torch.Tensor, eps: float = 1e-11):
    """Normalize data by subtracting mean and dividing by standard deviation."""
    mean = data.mean()
    std = data.std()
    return (data - mean) / (std + eps), mean, std

