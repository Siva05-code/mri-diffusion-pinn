"""
Visualization Utilities for MRI Image Reconstructions.

Generates side-by-side plots of ground truth, undersampled k-space, reconstructed images, and error maps.
"""

import os
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
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    # 1. Ground Truth Reference
    im0 = axes[0].imshow(gt, cmap="gray")
    axes[0].set_title("Ground Truth (Target)", fontsize=11)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Undersampling Mask
    im1 = axes[1].imshow(mask, cmap="gray", aspect="auto")
    axes[1].set_title(f"{accel_factor}x Undersampling Mask", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Undersampled K-Space Log Magnitude
    kspace_log = np.log(np.abs(undersampled_kspace) + 1e-9)
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

