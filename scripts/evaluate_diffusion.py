#!/usr/bin/env python3
"""
Evaluation Script for Conditional Diffusion MRI Reconstruction Baseline.

Evaluates standalone conditional diffusion on accelerated fastMRI slices:
- Zero-Filled (baseline)
- Conditional Diffusion (100-step DDPM reverse sampling from random Gaussian noise)

Supports deterministic reproducible evaluation using --seed.

Usage:
    # 1. Single-slice or validation subset evaluation:
    python scripts/evaluate_diffusion.py --config configs/base.yaml --accel 4 --seed 42

    # 2. Specific checkpoint evaluation:
    python scripts/evaluate_diffusion.py --checkpoint checkpoints/diffusion_accel4_best.pt --accel 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from src.data.preprocessing import align_and_normalize_slice, center_crop
from src.data.undersampling import RandomMaskFunc
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
    mask_func: RandomMaskFunc,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, any]:
    """Evaluates diffusion reconstruction on a single slice."""
    mask = mask_func(kspace.shape, seed=seed).to(device)
    undersampled = kspace * mask

    zf_complex = kspace_to_image_torch(undersampled)
    zf_mag = torch.abs(zf_complex)
    target_mag = torch.abs(target) if torch.is_complex(target) else target

    zf_norm, target_norm = align_and_normalize_slice(zf_mag, target_mag)

    condition = zf_norm.unsqueeze(0).unsqueeze(0).float().to(device)
    target_tensor = target_norm.unsqueeze(0).unsqueeze(0).float().to(device)

    # Deterministic DDPM reverse diffusion from seeded noise
    with torch.no_grad():
        diffusion_recon = schedule.sample(
            model=model,
            condition=condition,
            device=device,
        )

    zf_np = condition[0, 0].cpu().numpy().astype(np.float32)
    diff_np = diffusion_recon[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    tgt_np = target_tensor[0, 0].cpu().numpy().astype(np.float32)

    return {
        "zf_np": zf_np,
        "diff_np": diff_np,
        "target_np": tgt_np,
        "zf_metrics": {
            "psnr": calculate_psnr(tgt_np, zf_np),
            "ssim": calculate_ssim(tgt_np, zf_np),
            "nmse": calculate_nmse(tgt_np, zf_np),
        },
        "diff_metrics": {
            "psnr": calculate_psnr(tgt_np, diff_np),
            "ssim": calculate_ssim(tgt_np, diff_np),
            "nmse": calculate_nmse(tgt_np, diff_np),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Conditional Diffusion MRI Reconstruction")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--accel", type=int, default=4, help="Acceleration factor (4 or 8)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic evaluation")
    parser.add_argument("--file_path", type=str, default=None, help="Direct path to HDF5 file")
    parser.add_argument("--slice_idx", type=int, default=18, help="Slice index to evaluate")
    parser.add_argument("--max_samples", type=int, default=10, help="Max slices to evaluate if running on dataset")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
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
    print(f"Seed            : {args.seed} (Deterministic mode enabled)")
    print(f"Timesteps       : {timesteps}")
    print(f"Device          : {device}")
    print(f"Checkpoint      : {ckpt_path or 'Untrained (random initialization for verification)'}")
    print("=" * 65)

    model = ConditionalDiffusionUNet(base_channels=32, time_dim=128).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("[INFO] Running with model weights as initialized.")

    model.eval()
    schedule = DiffusionSchedule(timesteps=timesteps, device=device)
    mask_func = RandomMaskFunc(center_fractions=[center_frac], accelerations=[accel], seed=args.seed)

    results_dir = Path("results")
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Check if we should evaluate on dataset or single slice
    if args.file_path or (not val_dir or not Path(val_dir).exists() or len(list(Path(val_dir).glob("*.h5"))) == 0):
        kspace, target, source_desc = load_or_simulate_slice(args.file_path, args.slice_idx, val_dir)
        print(f"Evaluating on slice: {source_desc}")

        eval_res = evaluate_single_slice(
            kspace=kspace.to(device),
            target=target.to(device),
            model=model,
            schedule=schedule,
            mask_func=mask_func,
            device=device,
            seed=args.seed,
        )

        zf_m = eval_res["zf_metrics"]
        diff_m = eval_res["diff_metrics"]

        print("\n" + "=" * 65)
        print(f"RESULTS FOR {source_desc} ({accel}x Acceleration):")
        print("=" * 65)
        print(f"Zero-Filled : PSNR = {zf_m['psnr']:.2f} dB | SSIM = {zf_m['ssim']:.4f} | NMSE = {zf_m['nmse']:.4f}")
        print(f"Diffusion   : PSNR = {diff_m['psnr']:.2f} dB | SSIM = {diff_m['ssim']:.4f} | NMSE = {diff_m['nmse']:.4f}")
        print("=" * 65)

        # Plot figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(eval_res["zf_np"], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"Zero-Filled ({accel}x)\nPSNR: {zf_m['psnr']:.2f} dB | SSIM: {zf_m['ssim']:.4f}")
        axes[0].axis("off")

        axes[1].imshow(eval_res["diff_np"], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Conditional Diffusion\nPSNR: {diff_m['psnr']:.2f} dB | SSIM: {diff_m['ssim']:.4f}")
        axes[1].axis("off")

        axes[2].imshow(eval_res["target_np"], cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Ground Truth Target")
        axes[2].axis("off")

        plt.tight_layout()
        fig_path = results_dir / "figures" / f"diffusion_eval_accel{accel}_slice.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization to: {fig_path}")

    else:
        # Multi-slice validation dataset evaluation
        dataset = FastMRIDataset(val_dir, max_files=10)
        n_eval = min(len(dataset), args.max_samples)
        print(f"Evaluating on {n_eval} validation slices...")

        records = []
        for idx in range(n_eval):
            sample = dataset[idx]
            kspace_t = sample["kspace"].to(device)
            raw_tgt = sample.get("target")
            tgt_t = raw_tgt.to(device) if raw_tgt is not None else None

            slice_seed = args.seed + idx
            eval_res = evaluate_single_slice(
                kspace=kspace_t,
                target=tgt_t,
                model=model,
                schedule=schedule,
                mask_func=mask_func,
                device=device,
                seed=slice_seed,
            )

            records.append({
                "slice": idx,
                "zf_psnr": eval_res["zf_metrics"]["psnr"],
                "zf_ssim": eval_res["zf_metrics"]["ssim"],
                "zf_nmse": eval_res["zf_metrics"]["nmse"],
                "diff_psnr": eval_res["diff_metrics"]["psnr"],
                "diff_ssim": eval_res["diff_metrics"]["ssim"],
                "diff_nmse": eval_res["diff_metrics"]["nmse"],
            })

        df = pd.DataFrame(records)
        csv_path = results_dir / "tables" / f"diffusion_eval_accel{accel}.csv"
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 65)
        print(f"DIFFUSION BENCHMARK ({accel}x, N={n_eval} slices, seed={args.seed}):")
        print("=" * 65)
        print(f"Zero-Filled: PSNR {df['zf_psnr'].mean():.2f} ± {df['zf_psnr'].std():.2f} dB | "
              f"SSIM {df['zf_ssim'].mean():.4f} ± {df['zf_ssim'].std():.4f} | "
              f"NMSE {df['zf_nmse'].mean():.4f} ± {df['zf_nmse'].std():.4f}")
        print(f"Diffusion  : PSNR {df['diff_psnr'].mean():.2f} ± {df['diff_psnr'].std():.2f} dB | "
              f"SSIM {df['diff_ssim'].mean():.4f} ± {df['diff_ssim'].std():.4f} | "
              f"NMSE {df['diff_nmse'].mean():.4f} ± {df['diff_nmse'].std():.4f}")
        print("=" * 65)
        print(f"Saved slice metrics to: {csv_path}")


if __name__ == "__main__":
    main()