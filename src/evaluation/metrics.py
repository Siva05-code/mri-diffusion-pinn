"""
Evaluation Metrics for MRI Reconstruction.

Calculates Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM),
Normalized Mean Squared Error (NMSE), and K-Space Data Consistency Error.
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = None) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB."""
    if data_range is None:
        data_range = gt.max() - gt.min()
    return float(peak_signal_noise_ratio(gt, pred, data_range=data_range))


def calculate_ssim(gt: np.ndarray, pred: np.ndarray, data_range: float = None) -> float:
    """Calculate Structural Similarity Index (SSIM)."""
    if data_range is None:
        data_range = gt.max() - gt.min()
    return float(structural_similarity(gt, pred, data_range=data_range))


def calculate_nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate Normalized Mean Squared Error (NMSE).
    NMSE = ||gt - pred||_2^2 / ||gt||_2^2
    """
    return float(np.sum((gt - pred) ** 2) / (np.sum(gt ** 2) + 1e-12))


def calculate_kspace_error(y_measured: torch.Tensor, x_pred_complex: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Calculate k-space data-consistency error at acquired sampling locations.
    
    Error = || Mask * (FFT(x_pred) - y_measured) ||_2 / || y_measured ||_2
    """
    from src.physics.fft import image_to_kspace_torch
    
    if not torch.is_complex(x_pred_complex):
        x_pred_complex = x_pred_complex.to(torch.complex64)
        
    y_pred = mask * image_to_kspace_torch(x_pred_complex)
    diff = mask * (y_pred - y_measured)
    
    error = torch.norm(diff) / (torch.norm(mask * y_measured) + 1e-12)
    return float(error.item())


def evaluate_reconstruction(gt: np.ndarray, pred: np.ndarray, y_measured: torch.Tensor = None, x_pred_complex: torch.Tensor = None, mask: torch.Tensor = None) -> dict:
    """
    Compute a dictionary of all evaluation metrics for a single image slice.
    """
    data_range = gt.max() - gt.min()
    metrics = {
        "psnr": calculate_psnr(gt, pred, data_range=data_range),
        "ssim": calculate_ssim(gt, pred, data_range=data_range),
        "nmse": calculate_nmse(gt, pred),
    }
    
    if y_measured is not None and x_pred_complex is not None and mask is not None:
        metrics["kspace_error"] = calculate_kspace_error(y_measured, x_pred_complex, mask)
    else:
        metrics["kspace_error"] = 0.0

    return metrics

