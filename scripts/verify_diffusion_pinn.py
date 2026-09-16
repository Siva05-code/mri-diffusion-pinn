#!/usr/bin/env python3
"""
Verification Script for Physics-Informed Diffusion-PINN Pipeline.

Verifies end-to-end execution on a single fastMRI slice:
1. Loads raw fastMRI k-space slice (or simulated fastMRI-dimensioned slice if offline).
2. Applies Cartesian undersampling mask.
3. Computes zero-filled IFFT reconstruction and aligns to FOV.
4. Executes Conditional Diffusion pre-reconstruction -> Coarse image.
5. Executes PINN refinement -> Final image.
6. Computes all loss components:
   - Image loss (L1)
   - K-space loss (Fourier fidelity)
   - Physics loss (Data consistency + Gradient smoothness)
   - Total composite loss
7. Verifies gradient computation across both Diffusion and PINN modules.
8. Prints required diagnostic shapes and loss values.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.diffusion_pinn import DiffusionPINN
from src.physics.fft import kspace_to_image_torch, image_to_kspace_torch
from src.data.preprocessing import align_and_normalize_slice, center_crop
from src.data.undersampling import RandomMaskFunc
from src.losses.total_loss import TotalReconstructionLoss


def find_fastmri_h5_file(cli_path=None, config_path="configs/base.yaml"):
    candidates = []
    if cli_path:
        p = Path(cli_path)
        if p.is_file():
            return p
        if p.is_dir():
            candidates.append(p)

    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        val_dir = cfg.get("data", {}).get("val_dir")
        if val_dir:
            candidates.append(Path(val_dir))
        root_dir = cfg.get("data", {}).get("dataset_root")
        if root_dir:
            candidates.append(Path(root_dir))

    candidates.append(Path("data/fastmri/knee_singlecoil_val"))
    candidates.append(Path("data/fastmri"))

    for cdir in candidates:
        if cdir.exists():
            h5_files = sorted(list(cdir.glob("*.h5")) + list(cdir.glob("*.hdf5")) + list(cdir.glob("**/*.h5")))
            if h5_files:
                return h5_files[0]
    return None


def run_single_slice_verification(file_path: Optional[str] = None, config_path: str = "configs/base.yaml"):
    print("=" * 60)
    print("DIFFUSION + PINN SINGLE-SLICE PIPELINE VERIFICATION")
    print("=" * 60)

    # Resolve HDF5 file
    h5_file = None
    if file_path:
        p = Path(file_path)
        if p.exists():
            h5_file = p

    if h5_file is None:
        try:
            h5_file = find_fastmri_h5_file(config_path=config_path)
        except Exception:
            h5_file = None

    if h5_file is not None:
        print(f"[DATA] Loading real fastMRI file: {h5_file.resolve()}")
        with h5py.File(h5_file, "r") as hf:
            kspace_data = hf["kspace"]
            total_slices = kspace_data.shape[0]
            slice_idx = total_slices // 2
            raw_kspace_np = kspace_data[slice_idx]
            raw_target_np = hf["reconstruction_rss"][slice_idx] if "reconstruction_rss" in hf else None

        kspace_raw = torch.from_numpy(raw_kspace_np)
        if not torch.is_complex(kspace_raw):
            if kspace_raw.shape[-1] == 2:
                kspace_raw = torch.complex(kspace_raw[..., 0], kspace_raw[..., 1])
            else:
                kspace_raw = kspace_raw.to(torch.complex64)

        if raw_target_np is not None:
            target_raw = torch.from_numpy(raw_target_np.astype(np.float32))
        else:
            target_raw = torch.abs(kspace_to_image_torch(kspace_raw))
    else:
        print("[DATA] Official fastMRI HDF5 file not found locally (Kaggle dataset location).")
        print("[DATA] Generating single test slice with standard fastMRI dimensions (640, 368) for local verification.")
        # Create standard fastMRI test phantom
        torch.manual_seed(42)
        H, W = 640, 368
        sim_image = torch.zeros(H, W, dtype=torch.float32)
        sim_image[160:480, 24:344] = torch.rand(320, 320)
        kspace_raw = image_to_kspace_torch(sim_image.to(torch.complex64))
        target_raw = sim_image[160:480, 24:344]  # 320x320 ground truth target

    # 1. Raw k-space shape
    print(f"Raw k-space shape: {tuple(kspace_raw.shape)}")

    # 2. Undersampling mask
    mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4], seed=42)
    mask = mask_func(kspace_raw.shape, seed=42)
    print(f"Mask shape: {tuple(mask.shape)}")

    # 3. Undersampled k-space
    undersampled_kspace = kspace_raw * mask
    print(f"Undersampled k-space shape: {tuple(undersampled_kspace.shape)}")

    # 4. Zero-filled image
    zf_complex = kspace_to_image_torch(undersampled_kspace)
    zf_mag = torch.abs(zf_complex)
    zf_norm, target_norm = align_and_normalize_slice(zf_mag, target_raw, crop_size=(320, 320))

    zf_tensor = zf_norm.unsqueeze(0).unsqueeze(0).float()
    target_tensor = target_norm.unsqueeze(0).unsqueeze(0).float()
    print(f"Zero-filled image shape: {tuple(zf_tensor.shape)}")

    # 5. Initialize Diffusion + PINN model
    model = DiffusionPINN(
        diffusion_base_channels=16,
        diffusion_time_dim=64,
        pinn_base_channels=16,
        pinn_num_blocks=2,
        timesteps=10,
        enforce_dc=True,
    )

    # 6. Forward pass through Diffusion + PINN
    # Align k-space in 320x320 reconstruction domain
    k_gt_320 = image_to_kspace_torch(target_tensor.squeeze(1).to(torch.complex64))
    mask_320 = center_crop(mask, (1, 320))
    y_meas_320 = k_gt_320 * mask_320

    out = model.forward_train(
        target=target_tensor,
        condition=zf_tensor,
        y_measured=y_meas_320,
        mask=mask_320,
    )

    coarse_image = out["coarse"]
    final_pinn_image = out["refined"]

    print(f"Coarse diffusion image shape: {tuple(coarse_image.shape)}")
    print(f"Final PINN image shape: {tuple(final_pinn_image.shape)}")

    # 7. Losses
    criterion = TotalReconstructionLoss(
        image_loss_type="l1",
        lambda_image=1.0,
        lambda_kspace=1.0,
        lambda_physics=1.0,
        lambda_gradient=0.1,
        relative_kspace=True,
    )

    losses = criterion(
        i_pred=final_pinn_image,
        i_gt=target_tensor,
        y_measured=y_meas_320,
        k_gt=k_gt_320,
        mask=mask_320,
    )

    print(f"Image loss: {losses['image_loss'].item():.6f}")
    print(f"K-space loss: {losses['kspace_loss'].item():.6f}")
    print(f"Physics loss: {losses['physics_loss'].item():.6f}")
    print(f"Total loss: {losses['total_loss'].item():.6f}")

    # 8. Gradient verification
    losses["total_loss"].backward()
    diff_grad = any(p.grad is not None for p in model.diffusion.parameters())
    pinn_grad = any(p.grad is not None for p in model.pinn.parameters())

    assert diff_grad, "Gradient failed to reach diffusion module"
    assert pinn_grad, "Gradient failed to reach PINN module"
    print("Joint gradient backpropagation: Verified (Diffusion & PINN)")

    print("\n" + "=" * 60)
    print("DIFFUSION-PINN PIPELINE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Diffusion+PINN pipeline on single slice")
    parser.add_argument("--file", type=str, default=None, help="Path to fastMRI HDF5 file")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config")
    args = parser.parse_args()

    run_single_slice_verification(file_path=args.file, config_path=args.config)
