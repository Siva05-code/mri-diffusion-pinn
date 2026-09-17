#!/usr/bin/env python3
"""
Evaluation Script for Conditional Diffusion MRI Reconstruction Baseline.

Evaluates standalone conditional diffusion on accelerated fastMRI slices:
- Zero-Filled (baseline)
- Conditional Diffusion (100-step DDPM reverse sampling with raw-domain measurement guidance)

Supports deterministic reproducible evaluation using --seed and reports mean ± standard deviation
over the specified validation population.

Usage:
    # 1. Evaluate on validation subset with 4x acceleration:
    python scripts/evaluate_diffusion.py --config configs/base.yaml --accel 4 --seed 42

    # 2. Specific checkpoint evaluation:
    python scripts/evaluate_diffusion.py --checkpoint checkpoints/diffusion_accel4_best.pt --accel 4

    # 3. Legacy 2-channel checkpoint migration:
    python scripts/evaluate_diffusion.py --checkpoint checkpoints/legacy_model.pt --migrate_2ch --accel 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import FastMRIDataset
from src.data.preprocessing import normalize_slice_pair, center_crop
from src.data.undersampling import (
    EquispacedMaskFunc,
    RandomMaskFunc,
    create_mask_func,
    get_recon_conditioning_mask,
)
from src.evaluation.metrics import calculate_nmse, calculate_psnr, calculate_ssim
from src.models.diffusion import ConditionalDiffusionUNet
from src.models.diffusion_schedule import DiffusionSchedule
from src.physics.fft import image_to_kspace_torch, kspace_to_image_torch


def load_yaml(path: str) -> dict:
    if Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_or_simulate_slice(
    file_path: Optional[str] = None,
    slice_idx: int = 18,
    val_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Loads a fastMRI slice if available; otherwise falls back to a realistic knee simulation.
    """
    target_file = None
    if file_path and Path(file_path).exists():
        target_file = Path(file_path)
    elif val_dir and Path(val_dir).exists():
        files = sorted(list(Path(val_dir).glob("*.h5")))
        if files:
            target_file = files[0]

    if target_file is not None:
        try:
            with h5py.File(target_file, "r") as hf:
                total_slices = hf["kspace"].shape[0]
                idx = min(slice_idx, total_slices - 1)
                kspace_raw = hf["kspace"][idx]
                target_raw = hf["reconstruction_rss"][idx] if "reconstruction_rss" in hf else None

            kspace_t = torch.from_numpy(kspace_raw)
            if not torch.is_complex(kspace_t):
                if kspace_t.shape[-1] == 2:
                    kspace_t = torch.complex(kspace_t[..., 0], kspace_t[..., 1])
                else:
                    kspace_t = kspace_t.to(torch.complex64)

            if target_raw is not None:
                target_t = torch.from_numpy(target_raw.astype(np.float32))
            else:
                target_t = torch.abs(kspace_to_image_torch(kspace_t))

            return kspace_t, target_t, f"{target_file.name} (slice {idx})"
        except Exception as e:
            print(f"[WARN] Error reading {target_file}: {e}. Falling back to simulation.")

    # Numerical knee phantom on standard fastMRI [640, 368] grid
    print("[DATA] Generating realistic fastMRI knee phantom on [640, 368] grid for evaluation.")
    H, W = 640, 368
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    knee = (x**2 / 0.6**2 + y**2 / 0.8**2) < 0.8
    bone = ((x + 0.1)**2 / 0.2**2 + (y + 0.1)**2 / 0.3**2) < 0.5
    sim_img = torch.zeros(H, W, dtype=torch.float32)
    sim_img[knee] = 0.6 + 0.3 * torch.cos(8 * x[knee])
    sim_img[bone] = 0.95
    phase = 0.5 * torch.sin(3 * x) + 0.2 * torch.cos(2 * y)
    sim_complex = sim_img.to(torch.complex64) * torch.exp(1j * phase)
    kspace_t = image_to_kspace_torch(sim_complex)
    target_t = center_crop(sim_img, (320, 320))
    return kspace_t, target_t, "Numerical Knee Simulation [640, 368]"


def evaluate_single_slice(
    kspace: torch.Tensor,
    target: torch.Tensor,
    model: ConditionalDiffusionUNet,
    schedule: DiffusionSchedule,
    mask_func: Union[EquispacedMaskFunc, RandomMaskFunc],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, any]:
    """Evaluates diffusion reconstruction on a single slice using the unified physical pipeline."""
    # 1. Raw mask and physical measurement
    mask_raw = mask_func(kspace.shape, seed=seed).to(device)
    y_raw = kspace.to(device) * mask_raw

    # 2. Zero-filled reconstruction via centered IFFT
    zf_complex_raw = kspace_to_image_torch(y_raw)
    zf_complex_crop = center_crop(zf_complex_raw, (320, 320))
    zf_mag_crop = torch.abs(zf_complex_crop)
    phase_zf = torch.angle(zf_complex_crop)

    # 3. Ground truth target
    target_crop = center_crop(target.to(device), (320, 320))
    if torch.is_complex(target_crop):
        target_crop = torch.abs(target_crop)

    # 4. Shared normalization
    zf_norm, target_norm, scale = normalize_slice_pair(zf_mag_crop, target_crop, percentile=99.0)
    y_raw_norm = y_raw / scale

    condition = zf_norm.unsqueeze(0).unsqueeze(0).float().to(device)
    target_tensor = target_norm.unsqueeze(0).unsqueeze(0).float().to(device)
    phase_tensor = phase_zf.unsqueeze(0).unsqueeze(0).float().to(device)

    # 5. Conditioning mask for model
    mask_recon = get_recon_conditioning_mask(mask_raw, recon_size=(320, 320)).to(device)

    # Deterministic measurement-guided reverse diffusion from seeded noise
    with torch.no_grad():
        diffusion_recon = schedule.sample(
            model=model,
            condition=condition,
            mask=mask_recon,
            y_raw=y_raw_norm,
            mask_raw=mask_raw,
            phase_zf=phase_tensor,
            measurement_guided=True,
            device=device,
        )

    zf_np = condition[0, 0].cpu().numpy().astype(np.float32)
    diff_np = diffusion_recon[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    tgt_np = target_tensor[0, 0].cpu().numpy().astype(np.float32)

    data_range = float(tgt_np.max() - tgt_np.min() + 1e-12)

    return {
        "zf_np": zf_np,
        "diff_np": diff_np,
        "target_np": tgt_np,
        "zf_metrics": {
            "psnr": calculate_psnr(tgt_np, zf_np, data_range=data_range),
            "ssim": calculate_ssim(tgt_np, zf_np, data_range=data_range),
            "nmse": calculate_nmse(tgt_np, zf_np),
        },
        "diff_metrics": {
            "psnr": calculate_psnr(tgt_np, diff_np, data_range=data_range),
            "ssim": calculate_ssim(tgt_np, diff_np, data_range=data_range),
            "nmse": calculate_nmse(tgt_np, diff_np),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Conditional Diffusion MRI Reconstruction")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--accel", type=int, default=4, help="Acceleration factor (4 or 8)")
    parser.add_argument("--mask_type", type=str, default="equispaced", choices=["equispaced", "random"])
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic evaluation")
    parser.add_argument("--file_path", type=str, default=None, help="Direct path to HDF5 file")
    parser.add_argument("--slice_idx", type=int, default=18, help="Slice index to evaluate")
    parser.add_argument("--max_samples", type=int, default=10, help="Max slices to evaluate if running on dataset")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument(
        "--migrate_2ch",
        action="store_true",
        help="Explicitly migrate legacy 2-channel checkpoint by zero-initializing mask channel.",
    )
    args = parser.parse_args()

    # Set seeds for reproducible evaluation
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    val_dir = data_cfg.get("val_dir")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    accel = args.accel
    timesteps = cfg.get("diffusion", {}).get("timesteps", 100)
    center_frac = 0.08 if accel == 4 else 0.04

    # Determine checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        best_candidate = Path(f"checkpoints/diffusion_accel{accel}_best.pt")
        latest_candidate = Path(f"checkpoints/diffusion_accel{accel}_latest.pt")
        if best_candidate.exists():
            ckpt_path = str(best_candidate)
        elif latest_candidate.exists():
            ckpt_path = str(latest_candidate)
        else:
            ckpt_path = None

    print("=" * 65)
    print("CONDITIONAL DIFFUSION MRI EVALUATION")
    print("=" * 65)
    print(f"Acceleration    : {accel}x")
    print(f"Mask Protocol   : {args.mask_type.capitalize()} Cartesian")
    print(f"Seed            : {args.seed} (Deterministic mode enabled)")
    print(f"Timesteps       : {timesteps}")
    print(f"Device          : {device}")
    print(f"Checkpoint      : {ckpt_path or 'Untrained (random initialization for verification)'}")
    print("=" * 65)

    # Initialize model
    model = ConditionalDiffusionUNet(in_channels=3, base_channels=32, time_dim=128).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_compatible_state_dict(state_dict, migrate_2ch=args.migrate_2ch)
        print(f"[MODEL] Successfully loaded checkpoint weights from: {ckpt_path}")
    else:
        print("[MODEL] Running with initialized weights (for pipeline verification).")
    model.eval()

    schedule = DiffusionSchedule(timesteps=timesteps, device=device)
    mask_func = create_mask_func(
        mask_type=args.mask_type,
        center_fraction=center_frac,
        acceleration=accel,
        seed=args.seed,
    )

    # Multi-slice evaluation if validation directory exists
    if val_dir and Path(val_dir).exists() and not args.file_path:
        dataset = FastMRIDataset(val_dir, max_files=args.max_samples)
        if len(dataset) > 0:
            print(f"[EVAL] Evaluating across {len(dataset)} validation slices from {val_dir}...")
            zf_psnrs, zf_ssims, zf_nmses = [], [], []
            diff_psnrs, diff_ssims, diff_nmses = [], [], []

            for i in range(len(dataset)):
                sample = dataset[i]
                slice_seed = args.seed + i
                res = evaluate_single_slice(
                    kspace=sample["kspace"],
                    target=sample.get("target"),
                    model=model,
                    schedule=schedule,
                    mask_func=mask_func,
                    device=device,
                    seed=slice_seed,
                )
                zf_psnrs.append(res["zf_metrics"]["psnr"])
                zf_ssims.append(res["zf_metrics"]["ssim"])
                zf_nmses.append(res["zf_metrics"]["nmse"])
                diff_psnrs.append(res["diff_metrics"]["psnr"])
                diff_ssims.append(res["diff_metrics"]["ssim"])
                diff_nmses.append(res["diff_metrics"]["nmse"])

            print("\n" + "=" * 65)
            print(f"POPULATION EVALUATION RESULTS ({len(dataset)} slices)")
            print("=" * 65)
            print(f"Zero-Filled Baseline : PSNR: {np.mean(zf_psnrs):.2f} ± {np.std(zf_psnrs):.2f} dB | SSIM: {np.mean(zf_ssims):.4f} ± {np.std(zf_ssims):.4f} | NMSE: {np.mean(zf_nmses):.4f}")
            print(f"Conditional Diffusion: PSNR: {np.mean(diff_psnrs):.2f} ± {np.std(diff_psnrs):.2f} dB | SSIM: {np.mean(diff_ssims):.4f} ± {np.std(diff_ssims):.4f} | NMSE: {np.mean(diff_nmses):.4f}")
            print("=" * 65)
            return

    # Single slice evaluation
    kspace, target, source_name = load_or_simulate_slice(
        file_path=args.file_path,
        slice_idx=args.slice_idx,
        val_dir=val_dir,
    )
    res = evaluate_single_slice(
        kspace=kspace,
        target=target,
        model=model,
        schedule=schedule,
        mask_func=mask_func,
        device=device,
        seed=args.seed,
    )

    zf_m = res["zf_metrics"]
    diff_m = res["diff_metrics"]
    print("\n" + "=" * 65)
    print(f"SINGLE SLICE EVALUATION: {source_name}")
    print("=" * 65)
    print(f"Zero-Filled Baseline : PSNR: {zf_m['psnr']:.2f} dB | SSIM: {zf_m['ssim']:.4f} | NMSE: {zf_m['nmse']:.4f}")
    print(f"Conditional Diffusion: PSNR: {diff_m['psnr']:.2f} dB | SSIM: {diff_m['ssim']:.4f} | NMSE: {diff_m['nmse']:.4f}")
    print("=" * 65)

    # Save visual comparison
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"diffusion_eval_accel{accel}_slice.png"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(res["target_np"], cmap="gray")
    axes[0].set_title("Ground Truth Target")
    axes[0].axis("off")

    axes[1].imshow(res["zf_np"], cmap="gray")
    axes[1].set_title(f"Zero-Filled ({zf_m['psnr']:.2f} dB)")
    axes[1].axis("off")

    axes[2].imshow(res["diff_np"], cmap="gray")
    axes[2].set_title(f"Conditional Diffusion ({diff_m['psnr']:.2f} dB)")
    axes[2].axis("off")

    err = np.abs(res["diff_np"] - res["target_np"])
    im = axes[3].imshow(err, cmap="hot")
    axes[3].set_title("Absolute Error Map")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[OUTPUT] Saved figure: {fig_path}")


if __name__ == "__main__":
    main()