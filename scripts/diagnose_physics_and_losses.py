#!/usr/bin/env python3
"""
Diagnostic Script for MRI Fourier Consistency, Loss Scaling, and Physics Verification.

Computes exact quantitative diagnostic metrics:
A. Roundtrip FFT/IFFT reconstruction fidelity
B. Measured k-space energy
C. Predicted k-space energy
D. Zero-filled k-space consistency error
E. Data-consistency output k-space error
F. Image-domain L1 loss
G. Relative k-space loss

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
from src.physics.forward_operator import MRIForwardOperator
from src.physics.data_consistency import apply_data_consistency
from src.data.preprocessing import normalize_slice_pair, center_crop
from src.data.undersampling import (
    EquispacedMaskFunc,
    create_mask_func,
    get_recon_conditioning_mask,
)
from src.models.diffusion_pinn import DiffusionPINN
from src.losses.total_loss import TotalReconstructionLoss


def load_slice(file_path: str = None, config_path: str = "configs/base.yaml"):
    """Loads a slice from official fastMRI H5 if available, else generates standard knee phantom."""
    h5_file = None
    if file_path and Path(file_path).exists():
        h5_file = Path(file_path)
    else:
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

    # 1. Undersampling (4x equispaced)
    mask_func = create_mask_func(mask_type="equispaced", center_fraction=0.08, acceleration=4, seed=42)
    mask_raw = mask_func(kspace_raw.shape, seed=42)
    kspace_under = kspace_raw * mask_raw
    zf_full_complex = kspace_to_image_torch(kspace_under)
    zf_complex_crop = center_crop(zf_full_complex, (320, 320))
    zf_mag_crop = torch.abs(zf_complex_crop)
    phase_zf = torch.angle(zf_complex_crop)

    # 2. Shared Normalization to 320x320
    target_crop = center_crop(target_raw, (320, 320))
    zf_norm, target_norm, scale = normalize_slice_pair(zf_mag_crop, target_crop, percentile=99.0)
    y_raw_norm = kspace_under / scale

    # 3. Model Prediction (one pass through Diffusion + PINN)
    forward_op = MRIForwardOperator()
    mask_recon = get_recon_conditioning_mask(mask_raw, recon_size=(320, 320))

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
            mask=mask_recon,
            mask_raw=mask_raw,
            y_measured=y_raw_norm.unsqueeze(0),
            phase=phase_zf.unsqueeze(0).unsqueeze(0),
        )
        pred_img = out["refined"]  # [1, 1, 320, 320]

    # Data consistency
    pred_dc = forward_op.data_consistency(
        x_crop=pred_img.squeeze(0).squeeze(0),
        y_raw=y_raw_norm,
        mask_raw=mask_raw,
        phase=phase_zf,
        return_complex=False,
    )

    # Roundtrip FFT/IFFT errors
    img_full_complex = kspace_to_image_torch(kspace_raw)
    k_full_from_img = image_to_kspace_torch(img_full_complex)
    raw_err_fft_gt = torch.norm(k_full_from_img - kspace_raw).item()

    raw_meas_energy = torch.norm(kspace_under).item()
    norm_meas_energy = torch.norm(y_raw_norm).item()

    # Pred k-space
    k_pred_raw = forward_op.forward_kspace(pred_img.squeeze(), phase=phase_zf, target_shape=kspace_raw.shape)
    norm_pred_energy = torch.norm(k_pred_raw).item()

    # Consistency errors
    diff_pred = mask_raw * (k_pred_raw - y_raw_norm)
    dc_err = torch.norm(diff_pred).item() / (norm_meas_energy + 1e-12)

    k_dc = forward_op.forward_kspace(pred_dc, phase=phase_zf, target_shape=kspace_raw.shape)
    diff_dc = mask_raw * (k_dc - y_raw_norm)
    post_dc_err = torch.norm(diff_dc).item() / (norm_meas_energy + 1e-12)

    img_l1_loss = torch.mean(torch.abs(pred_img.squeeze() - target_norm)).item()

    print("\n--- DIAGNOSTIC VALUES ---")
    print(f"A. ||FFT(I_gt) - K_gt|| (Roundtrip Error on 640x368 grid): {raw_err_fft_gt:.2e}")
    print(f"B. Measured k-space energy (Raw / Normalized):            {raw_meas_energy:.2f} / {norm_meas_energy:.2f}")
    print(f"C. Predicted k-space energy (Normalized):                 {norm_pred_energy:.2f}")
    print(f"D. Pre-DC k-space relative error:                         {dc_err:.4f}")
    print(f"E. Post-DC k-space relative error:                        {post_dc_err:.2e}")
    print(f"F. Image L1 loss against target:                          {img_l1_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diagnostics on MRI physics pipeline")
    parser.add_argument("--file", type=str, default=None, help="Path to H5 file")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config")
    args = parser.parse_args()
    run_diagnostics(args.file, args.config)
