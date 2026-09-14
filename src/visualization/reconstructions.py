"""
Visualization Utilities for MRI Image Reconstructions.

Generates side-by-side plots of ground truth, undersampled k-space, reconstructed images, and error maps.
"""

import os
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_zero_filled_comparison(
    gt: np.ndarray,
    mask: np.ndarray,
    undersampled_kspace: np.ndarray,
    zf_reconstruction: np.ndarray,
    error_map: np.ndarray,
    metrics: dict,
    accel_factor: int = 4,
    save_path: str = None
):
    """
    Plot 5-panel figure: GT, Mask, K-space Log-Mag, Zero-Filled Recon, Error Map.

    Supports both 1D Cartesian masks (broadcast to 2D image shape) and already-2D masks.
    """
    # Convert inputs to numpy arrays if needed
    if hasattr(gt, "cpu"):
        gt = gt.cpu().numpy()
    gt = np.squeeze(np.asarray(gt))

    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    mask = np.squeeze(np.asarray(mask))

    # Convert 1D Cartesian phase-encoding mask to 2D matching image dimensions
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[None, :], gt.shape)
    elif mask.ndim > 2:
        mask = np.squeeze(mask)
        if mask.ndim == 1:
            mask = np.broadcast_to(mask[None, :], gt.shape)

    if hasattr(zf_reconstruction, "cpu"):
        zf_reconstruction = zf_reconstruction.cpu().numpy()
    zf_reconstruction = np.squeeze(np.asarray(zf_reconstruction))

    if hasattr(error_map, "cpu"):
        error_map = error_map.cpu().numpy()
    error_map = np.squeeze(np.asarray(error_map))

    if hasattr(undersampled_kspace, "cpu"):
        undersampled_kspace = undersampled_kspace.cpu().numpy()
    undersampled_kspace = np.squeeze(np.asarray(undersampled_kspace))

    # Compute k-space magnitude in log scale
    if np.iscomplexobj(undersampled_kspace):
        kspace_mag = np.abs(undersampled_kspace)
    elif undersampled_kspace.ndim == 3 and undersampled_kspace.shape[-1] == 2:
        kspace_mag = np.sqrt(undersampled_kspace[..., 0]**2 + undersampled_kspace[..., 1]**2)
    else:
        kspace_mag = np.abs(undersampled_kspace)
    kspace_log = np.log(kspace_mag + 1e-9)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    # 1. Ground Truth Reference
    im0 = axes[0].imshow(gt, cmap="gray")
    axes[0].set_title("Ground Truth (Target)", fontsize=11)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Undersampling Mask (guaranteed 2D)
    # 2. Undersampling Mask
    im1 = axes[1].imshow(mask, cmap="gray", aspect="auto")
    axes[1].set_title(f"{accel_factor}x Undersampling Mask", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Undersampled K-Space Log Magnitude
    im2 = axes[2].imshow(kspace_log, cmap="gray")
    axes[2].set_title(f"{accel_factor}x Undersampled K-Space", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Zero-Filled Reconstruction
    im3 = axes[3].imshow(zf_reconstruction, cmap="gray")
    title_str = f"Zero-Filled {accel_factor}x\nPSNR: {metrics['psnr']:.2f}dB | SSIM: {metrics['ssim']:.3f}"
    axes[3].set_title(title_str, fontsize=10)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. Error Map
    im4 = axes[4].imshow(error_map, cmap="magma")
    axes[4].set_title(f"Error Map (|GT - Recon|)\nNMSE: {metrics['nmse']:.4f}", fontsize=10)
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_unet_comparison(
    gt: np.ndarray,
    zf: np.ndarray,
    unet_pred: np.ndarray,
    error_map: np.ndarray,
    metrics: Dict[str, float],
    accel_factor: int = 4,
    zf_metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot 4-panel representative figure:
    1. Ground Truth Reference
    2. Zero-Filled Input
    3. U-Net Reconstruction
    4. Absolute Error Map (|GT - UNet|)
    """
    if hasattr(gt, "cpu"):
        gt = gt.cpu().numpy()
    gt = np.squeeze(np.asarray(gt))

    if hasattr(zf, "cpu"):
        zf = zf.cpu().numpy()
    zf = np.squeeze(np.asarray(zf))

    if hasattr(unet_pred, "cpu"):
        unet_pred = unet_pred.cpu().numpy()
    unet_pred = np.squeeze(np.asarray(unet_pred))

    if hasattr(error_map, "cpu"):
        error_map = error_map.cpu().numpy()
    error_map = np.squeeze(np.asarray(error_map))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # 1. Ground Truth
    im0 = axes[0].imshow(gt, cmap="gray")
    axes[0].set_title("Ground Truth Reference", fontsize=11)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Zero-Filled Input
    im1 = axes[1].imshow(zf, cmap="gray")
    zf_str = f"Zero-Filled Input ({accel_factor}x)"
    if zf_metrics and "psnr" in zf_metrics:
        zf_str += f"\nPSNR: {zf_metrics['psnr']:.2f}dB | SSIM: {zf_metrics['ssim']:.3f}"
    axes[1].set_title(zf_str, fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. U-Net Reconstruction
    im2 = axes[2].imshow(unet_pred, cmap="gray")
    title_str = f"U-Net Baseline ({accel_factor}x)\nPSNR: {metrics['psnr']:.2f}dB | SSIM: {metrics['ssim']:.3f}"
    axes[2].set_title(title_str, fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Error Map
    im3 = axes[3].imshow(error_map, cmap="magma")
    axes[3].set_title(f"Error Map (|GT - U-Net|)\nNMSE: {metrics['nmse']:.4f}", fontsize=10)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig
