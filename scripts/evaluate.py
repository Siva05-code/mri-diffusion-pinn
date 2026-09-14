#!/usr/bin/env python3
"""
CLI Evaluation Script for MRI Reconstruction: Zero-Filled vs U-Net Baseline.

Usage:
    # Evaluate on 4x acceleration using a trained checkpoint:
    python3 scripts/evaluate.py --config configs/base.yaml --accel 4 --checkpoint checkpoints/unet_accel4_best.pt --subset_files 5

    # Evaluate on 8x acceleration:
    python3 scripts/evaluate.py --config configs/base.yaml --accel 8 --checkpoint checkpoints/unet_accel8_best.pt --subset_files 5

    # Compare both 4x and 8x:
    python3 scripts/evaluate.py --config configs/base.yaml --accel all --subset_files 5
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import yaml

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.unet import UNet
from src.data.dataset import FastMRIDataset
from src.data.undersampling import RandomMaskFunc
from src.data.preprocessing import complex_abs, align_and_normalize_slice
from src.physics.fft import kspace_to_image_torch
from src.evaluation.metrics import calculate_psnr, calculate_ssim, calculate_nmse
from src.visualization.error_maps import compute_absolute_error_map
from src.visualization.reconstructions import plot_unet_comparison


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def evaluate_slice(
    model: Optional[torch.nn.Module],
    sample: dict,
    mask_func: RandomMaskFunc,
    seed: int,
    device: torch.device,
) -> Dict[str, any]:
    """
    Evaluates a single slice for both Zero-Filled and U-Net reconstruction.
    """
    kspace = sample["kspace"]
    slice_seed = seed + int(sample.get("slice_idx", 0))
    mask = mask_func(kspace.shape, seed=slice_seed)

    # 1. Undersample k-space
    y_undersampled = mask * kspace

    # 2. Zero-Filled reconstruction via centered IFFT
    zf_complex = kspace_to_image_torch(y_undersampled)
    zf_mag = complex_abs(zf_complex)

    # 3. Reference ground truth
    if "target" in sample and sample["target"] is not None:
        target_mag = sample["target"]
    else:
        gt_complex = kspace_to_image_torch(kspace)
        target_mag = complex_abs(gt_complex)

    # 4. Spatial dimension verification and consistent [0, 1] normalization
    zf_norm, target_norm = align_and_normalize_slice(zf_mag, target_mag)

    gt_np = target_norm.cpu().numpy()
    zf_np = zf_norm.cpu().numpy()

    # Metrics for Zero-Filled
    zf_metrics = {
        "psnr": calculate_psnr(gt_np, zf_np),
        "ssim": calculate_ssim(gt_np, zf_np),
        "nmse": calculate_nmse(gt_np, zf_np),
    }

    # Metrics for U-Net (if model loaded)
    unet_metrics = None
    unet_np = None
    if model is not None:
        x_input = zf_norm.unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = model(x_input)
        unet_np = pred.squeeze().cpu().numpy()
        unet_np = np.clip(unet_np, 0.0, 1.0)
        unet_metrics = {
            "psnr": calculate_psnr(gt_np, unet_np),
            "ssim": calculate_ssim(gt_np, unet_np),
            "nmse": calculate_nmse(gt_np, unet_np),
        }

    return {
        "gt": gt_np,
        "zf": zf_np,
        "unet": unet_np,
        "zf_metrics": zf_metrics,
        "unet_metrics": unet_metrics,
        "filename": sample.get("filename", ""),
        "slice_idx": sample.get("slice_idx", 0),
    }


def run_evaluation_for_acceleration(
    accel: int,
    val_dir: str,
    checkpoint_path: Optional[str],
    subset_files: Optional[int],
    seed: int,
    device: torch.device,
    output_dir: str,
) -> Dict[str, any]:
    """
    Runs evaluation for a specific acceleration factor (4x or 8x).
    """
    center_fraction = 0.08 if accel == 4 else 0.04
    mask_func = RandomMaskFunc(
        center_fractions=[center_fraction],
        accelerations=[accel],
        seed=seed,
    )

    dataset = FastMRIDataset(root_dir=val_dir, max_files=subset_files)
    if len(dataset) == 0:
        raise RuntimeError(f"No validation samples found in {val_dir}")

    # Load model if checkpoint provided
    model = None
    if checkpoint_path and Path(checkpoint_path).exists():
        model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded U-Net checkpoint from: {checkpoint_path}")
    else:
        print(f"No checkpoint provided or found at '{checkpoint_path}'. Evaluating Zero-Filled baseline only.")

    results_records = []
    first_sample_data = None

    print(f"\nEvaluating {len(dataset)} slices at {accel}x acceleration...")
    for idx in range(len(dataset)):
        sample = dataset[idx]
        eval_data = evaluate_slice(model, sample, mask_func, seed, device)

        if first_sample_data is None:
            first_sample_data = eval_data

        rec_zf = {
            "method": "Zero-Filled",
            "acceleration": accel,
            "filename": eval_data["filename"],
            "slice_idx": eval_data["slice_idx"],
            "psnr": eval_data["zf_metrics"]["psnr"],
            "ssim": eval_data["zf_metrics"]["ssim"],
            "nmse": eval_data["zf_metrics"]["nmse"],
        }
        results_records.append(rec_zf)

        if eval_data["unet_metrics"] is not None:
            rec_unet = {
                "method": "U-Net",
                "acceleration": accel,
                "filename": eval_data["filename"],
                "slice_idx": eval_data["slice_idx"],
                "psnr": eval_data["unet_metrics"]["psnr"],
                "ssim": eval_data["unet_metrics"]["ssim"],
                "nmse": eval_data["unet_metrics"]["nmse"],
            }
            results_records.append(rec_unet)

    # Save visual comparison for the first representative slice
    if first_sample_data is not None and first_sample_data["unet"] is not None:
        gt = first_sample_data["gt"]
        zf = first_sample_data["zf"]
        unet_img = first_sample_data["unet"]
        error_map = compute_absolute_error_map(gt, unet_img)

        fig_path = os.path.join(output_dir, "figures", f"unet_accel{accel}_sample.png")
        plot_unet_comparison(
            gt=gt,
            zf=zf,
            unet_pred=unet_img,
            error_map=error_map,
            metrics=first_sample_data["unet_metrics"],
            accel_factor=accel,
            zf_metrics=first_sample_data["zf_metrics"],
            save_path=fig_path,
        )
        print(f"Saved representative reconstruction figure to: {fig_path}")

    return {
        "accel": accel,
        "records": results_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Zero-Filled vs U-Net MRI Reconstruction Baseline.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base config YAML.")
    parser.add_argument("--accel", type=str, default="4", choices=["4", "8", "all"], help="Acceleration factor.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to U-Net model checkpoint.")
    parser.add_argument("--subset_files", type=int, default=None, help="Limit number of validation files.")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation directory.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu').")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    paths_cfg = cfg.get("paths", {})

    val_dir = args.val_dir or data_cfg.get("val_dir")
    if not val_dir or not Path(val_dir).exists():
        raise FileNotFoundError(f"Validation directory '{val_dir}' does not exist.")

    seed = exp_cfg.get("seed", 42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results_dir = paths_cfg.get("results_dir", "results")

    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

    accel_factors = [4, 8] if args.accel == "all" else [int(args.accel)]
    all_records = []

    for accel in accel_factors:
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            default_ckpt = Path(paths_cfg.get("checkpoints_dir", "checkpoints")) / f"unet_accel{accel}_best.pt"
            if default_ckpt.exists():
                ckpt_path = str(default_ckpt)

        res = run_evaluation_for_acceleration(
            accel=accel,
            val_dir=val_dir,
            checkpoint_path=ckpt_path,
            subset_files=args.subset_files,
            seed=seed,
            device=device,
            output_dir=results_dir,
        )
        all_records.extend(res["records"])

    # Aggregate tabular metrics
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(results_dir, "tables", "unet_baseline_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved slice metrics to: {csv_path}")

    # Compute and display summary statistics (mean ± std)
    summary_list = []
    print("\n========================================================")
    print("           QUANTITATIVE EVALUATION SUMMARY")
    print("========================================================")
    for (method, accel), group in df.groupby(["method", "acceleration"]):
        psnr_mean, psnr_std = group["psnr"].mean(), group["psnr"].std()
        ssim_mean, ssim_std = group["ssim"].mean(), group["ssim"].std()
        nmse_mean, nmse_std = group["nmse"].mean(), group["nmse"].std()

        print(f"[{method} {accel}x] (N={len(group)} slices):")
        print(f"  PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB")
        print(f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
        print(f"  NMSE: {nmse_mean:.4f} ± {nmse_std:.4f}")

        summary_list.append({
            "method": method,
            "acceleration": int(accel),
            "num_slices": len(group),
            "psnr_mean": round(float(psnr_mean), 3),
            "psnr_std": round(float(psnr_std), 3),
            "ssim_mean": round(float(ssim_mean), 4),
            "ssim_std": round(float(ssim_std), 4),
            "nmse_mean": round(float(nmse_mean), 5),
            "nmse_std": round(float(nmse_std), 5),
        })

    # Save JSON experiment summary
    json_path = os.path.join(results_dir, "logs", "unet_experiment.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "experiment": "unet_baseline",
                "validation_directory": val_dir,
                "subset_files": args.subset_files,
                "results": summary_list,
            },
            f,
            indent=2,
        )
    print(f"\nExperiment summary JSON saved to: {json_path}")


if __name__ == "__main__":
    main()

