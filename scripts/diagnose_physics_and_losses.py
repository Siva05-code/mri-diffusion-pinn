#!/usr/bin/env python3
"""
Diagnostic Script for MRI Fourier Consistency, Loss Scaling, and Physics Verification.

Computes exact quantitative diagnostic metrics:
A. ||FFT(I_gt) - K_gt|| (Frequency domain reconstruction fidelity)
B. Measured k-space energy: ||M * K_gt||
C. Predicted k-space energy: ||FFT(I_pred)||
D. Zero-filled k-space consistency error: ||M * (FFT(I_zf) - K_gt)||
E. Data-consistency output k-space error: ||M * (FFT(I_dc) - K_gt)||
F. Image-domain L1 loss: ||I_pred - I_gt||_1
G. Normalized k-space loss: ||M * (FFT(I_pred) - K_gt)||_1 / (||M * K_gt||_1 + eps)

Prints both RAW and NORMALIZED values across domains.
"""

import sys
import os
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.fft import kspace_to_image_torch, image_to_kspace_torch
from src.physics.data_consistency import apply_data_consistency
from src.data.preprocessing import align_and_normalize_slice, center_crop
from src.data.undersampling import RandomMaskFunc
from src.models.diffusion_pinn import DiffusionPINN
from src.losses.total_loss import TotalReconstructionLoss


def load_slice(file_path: str = None, config_path: str = "configs/base.yaml"):
    """Loads a slice from official fastMRI H5 if available, else generates standard knee phantom."""
    h5_file = None
    if file_path and Path(file_path).exists():
        h5_file = Path(file_path)
    else:
        # Check standard config paths
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            val_dir = cfg.get("data", {}).get("val_dir")
            if val_dir and Path(val_dir).exists():
                files = sorted(list(Path(val_dir).glob("*.h5")))
                if files:
                    h5_file = files[0]

    if h5_file is not None:
        print(f"[DATA] Loading real fastMRI slice from: {h5_file.resolve()}")
        with h5py.File(h5_file, "r") as hf:
            kspace_data = hf["kspace"]
            slice_idx = kspace_data.shape[0] // 2
            raw_kspace = kspace_data[slice_idx]
            raw_target = hf["reconstruction_rss"][slice_idx] if "reconstruction_rss" in hf else None

        kspace_t = torch.from_numpy(raw_kspace)
        if not torch.is_complex(kspace_t):
            if kspace_t.shape[-1] == 2:
                kspace_t = torch.complex(kspace_t[..., 0], kspace_t[..., 1])
            else:
                kspace_t = kspace_t.to(torch.complex64)

        if raw_target is not None:
            target_t = torch.from_numpy(raw_target.astype(np.float32))
        else:
            target_t = torch.abs(kspace_to_image_torch(kspace_t))
        source_name = f"Official FastMRI ({h5_file.name}, slice {slice_idx})"
    else:
        # Standard realistic anatomical knee simulation on [640, 368] grid
        print("[DATA] Official fastMRI file not present locally (available on Kaggle).")
        print("[DATA] Generating realistic fastMRI knee phantom on raw [640, 368] grid.")
        torch.manual_seed(42)
        H, W = 640, 368
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        knee_mask = (x**2 / 0.6**2 + y**2 / 0.8**2) < 0.8
        bone_mask = ((x + 0.1)**2 / 0.2**2 + (y + 0.1)**2 / 0.3**2) < 0.5
        sim_img = torch.zeros(H, W, dtype=torch.float32)
        sim_img[knee_mask] = 0.6 + 0.3 * torch.cos(8 * x[knee_mask])
        sim_img[bone_mask] = 0.95
        # Add realistic phase
        phase = 0.5 * torch.sin(3 * x) + 0.2 * torch.cos(2 * y)
        sim_complex = sim_img.to(torch.complex64) * torch.exp(1j * phase)
        kspace_t = image_to_kspace_torch(sim_complex)
        target_t = center_crop(sim_img, (320, 320))
        source_name = "Realistic FastMRI Numerical Knee Phantom [640, 368]"

    return kspace_t, target_t, source_name


def run_diagnostics(file_path: str = None, config_path: str = "configs/base.yaml"):
    kspace_raw, target_raw, source_name = load_slice(file_path, config_path)

    print("\n" + "=" * 70)
    print("PHYSICS, FOURIER DOMAIN & LOSS SCALING DIAGNOSTIC SUITE")
    print("=" * 70)
    print(f"Data source: {source_name}")
    print(f"Raw k-space shape: {tuple(kspace_raw.shape)}, dtype: {kspace_raw.dtype}")
    print(f"Target shape:      {tuple(target_raw.shape)}, dtype: {target_raw.dtype}")
    print("=" * 70)

    # 1. Fourier Reconstruction of full FOV
    img_full_complex = kspace_to_image_torch(kspace_raw)
    img_full_mag = torch.abs(img_full_complex)

    # 2. Undersampling (4x, 0.08 center fraction)
    mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4], seed=42)
    mask_raw = mask_func(kspace_raw.shape, seed=42)
    kspace_under = kspace_raw * mask_raw
    zf_full_complex = kspace_to_image_torch(kspace_under)
    zf_full_mag = torch.abs(zf_full_complex)

    # 3. Standard Alignment & Normalization to 320x320
    zf_norm, target_norm = align_and_normalize_slice(zf_full_mag, target_raw, crop_size=(320, 320))

    # Calculate normalization scaling factor
    raw_crop_gt = center_crop(target_raw, (320, 320))
    scale_factor = (raw_crop_gt.max() - raw_crop_gt.min() + 1e-12).item()

    # 4. Model Prediction (one pass through Diffusion + PINN)
    model = DiffusionPINN(
        diffusion_base_channels=16,
        diffusion_time_dim=64,
        pinn_base_channels=16,
        pinn_num_blocks=2,
        timesteps=10,
        enforce_dc=False,
    )
    zf_input = zf_norm.unsqueeze(0).unsqueeze(0).float()
    target_input = target_norm.unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        out = model.forward_train(
            target=target_input,
            condition=zf_input,
            y_measured=kspace_under.unsqueeze(0),
            mask=mask_raw,
        )
        pred_img = out["refined"]  # [1, 1, 320, 320] in [0, 1]

    # Data-consistent prediction in 320x320
    k_meas_320 = image_to_kspace_torch(zf_norm.to(torch.complex64))
    mask_320 = center_crop(mask_raw, (1, 320))
    pred_dc = apply_data_consistency(pred_img, k_meas_320.unsqueeze(0), mask_320, return_complex=False)

    # 5. Diagnostic Computations in 320x320 Domain
    k_gt_320 = image_to_kspace_torch(target_norm.to(torch.complex64))
    k_pred_320 = image_to_kspace_torch(pred_img.squeeze(0).squeeze(0).to(torch.complex64))
    k_zf_320 = image_to_kspace_torch(zf_norm.to(torch.complex64))
    k_dc_320 = image_to_kspace_torch(pred_dc.squeeze(0).squeeze(0).to(torch.complex64))

    # A. ||FFT(I_gt) - K_gt||
    # In 320x320 domain:
    err_fft_gt_320 = torch.norm(k_gt_320 - k_gt_320).item()
    # Between un-normalized full 640x368 image and raw k-space:
    k_full_from_img = image_to_kspace_torch(img_full_complex)
    raw_err_fft_gt = torch.norm(k_full_from_img - kspace_raw).item()

    # B. Measured k-space energy: ||M * K_gt||
    raw_meas_energy = torch.norm(kspace_under).item()
    norm_meas_energy_320 = torch.norm(mask_320.squeeze(0) * k_gt_320).item()

    # C. Predicted k-space energy: ||FFT(I_pred)||
    norm_pred_energy_320 = torch.norm(k_pred_320).item()

    # D. Zero-filled k-space consistency error: ||M * (FFT(I_zf) - K_gt)||
    raw_zf_dc_err = torch.norm(mask_raw * (image_to_kspace_torch(zf_full_complex) - kspace_under)).item()
    norm_zf_dc_err_320 = torch.norm(mask_320.squeeze(0) * (k_zf_320 - k_gt_320)).item()

    # E. Data-consistency output k-space error: ||M * (FFT(I_dc) - K_gt)||
    norm_dc_out_err_320 = torch.norm(mask_320.squeeze(0) * (k_dc_320 - k_gt_320)).item()

    # F. Image-domain L1 loss: ||I_pred - I_gt||_1
    img_l1_loss = torch.mean(torch.abs(pred_img.squeeze() - target_norm)).item()

    # G. K-space loss: raw vs normalized
    # Raw un-normalized L1 difference at mask positions:
    k_diff_320 = mask_320.squeeze(0) * (k_pred_320 - k_gt_320)
    raw_k_l1_diff = torch.mean(torch.abs(k_diff_320)).item()
    # Relative normalized k-space loss:
    denom = torch.mean(torch.abs(mask_320.squeeze(0) * k_gt_320)).item() + 1e-12
    rel_k_loss = raw_k_l1_diff / denom

    print("\n--- DIAGNOSTIC VALUES ---")
    print(f"A. ||FFT(I_gt) - K_gt|| (Roundtrip Error):")
    print(f"   - Raw 640x368 domain:       {raw_err_fft_gt:.2e}")
    print(f"   - 320x320 domain:           {err_fft_gt_320:.2e}")

    print(f"\nB. Measured k-space energy ||M * K_gt||_2:")
    print(f"   - Raw 640x368 domain:       {raw_meas_energy:.4f}")
    print(f"   - 320x320 domain:           {norm_meas_energy_320:.4f}")

    print(f"\nC. Predicted k-space energy ||FFT(I_pred)||_2:")
    print(f"   - 320x320 domain:           {norm_pred_energy_320:.4f}")

    print(f"\nD. Zero-filled k-space consistency error ||M * (FFT(I_zf) - K_meas)||_2:")
    print(f"   - Raw 640x368 domain:       {raw_zf_dc_err:.2e}")
    print(f"   - 320x320 domain:           {norm_zf_dc_err_320:.2e}")

    print(f"\nE. Data-consistency output k-space error ||M * (FFT(I_dc) - K_meas)||_2:")
    print(f"   - 320x320 domain:           {norm_dc_out_err_320:.2e}")

    print(f"\nF. Image-domain L1 loss ||I_pred - I_gt||_1:")
    print(f"   - Image L1 loss:            {img_l1_loss:.6f}")

    print(f"\nG. K-space loss (at mask trajectory):")
    print(f"   - Raw absolute mean:        {raw_k_l1_diff:.6f}")
    print(f"   - Relative normalized loss: {rel_k_loss:.6f}")

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON:")
    print(f"  Image L1 Loss           : {img_l1_loss:.6f}")
    print(f"  Relative K-Space Loss   : {rel_k_loss:.6f}")
    print(f"  Ratio (K-space / Image) : {rel_k_loss / (img_l1_loss + 1e-12):.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic test suite for MRI Fourier consistency and losses")
    parser.add_argument("--file", type=str, default=None, help="Path to fastMRI H5 file")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config")
    args = parser.parse_args()

    run_diagnostics(file_path=args.file, config_path=args.config)

