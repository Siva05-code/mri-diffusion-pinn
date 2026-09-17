#!/usr/bin/env python3
"""
Benchmark Evaluation Script: Zero-Filled vs U-Net vs Diffusion vs Proposed Diffusion+PINN.

Evaluates on the exact same fastMRI validation slices and reproducible sampling masks:
1. Zero-Filled (baseline)
2. U-Net (baseline)
3. Conditional Diffusion (pre-reconstruction baseline with raw measurement guidance)
4. Diffusion + PINN (proposed architecture with raw physics refinement)

Calculates:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- NMSE (Normalized Mean Squared Error)

Saves results to:
- results/tables/diffusion_pinn_benchmark_accel{accel}.csv
- results/logs/diffusion_pinn_benchmark_accel{accel}_summary.json
- results/figures/diffusion_pinn_accel{accel}_comparison.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

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
from src.models.diffusion_pinn import DiffusionPINN
from src.models.diffusion_schedule import DiffusionSchedule
from src.models.unet import UNet
from src.physics.fft import image_to_kspace_torch, kspace_to_image_torch


def load_yaml(path: str) -> dict:
    if Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def evaluate_slice_all_methods(
    sample: dict,
    mask_func: Union[EquispacedMaskFunc, RandomMaskFunc],
    seed: int,
    device: torch.device,
    unet_model: Optional[torch.nn.Module] = None,
    diffusion_model: Optional[torch.nn.Module] = None,
    diffusion_schedule: Optional[DiffusionSchedule] = None,
    diffusion_pinn_model: Optional[DiffusionPINN] = None,
) -> Dict[str, any]:
    """Evaluates a single fastMRI slice across all available reconstruction methods."""
    kspace = sample["kspace"].to(device)
    if not torch.is_complex(kspace):
        if kspace.shape[-1] == 2:
            kspace = torch.complex(kspace[..., 0], kspace[..., 1])
        else:
            kspace = kspace.to(torch.complex64)

    slice_seed = seed + int(sample.get("slice_idx", 0))
    mask_raw = mask_func(kspace.shape, seed=slice_seed).to(device)

    # 1. Physical undersampling
    y_raw = mask_raw * kspace

    # 2. Zero-Filled reconstruction via centered IFFT
    zf_complex_raw = kspace_to_image_torch(y_raw)
    zf_complex_crop = center_crop(zf_complex_raw, (320, 320))
    zf_mag_crop = torch.abs(zf_complex_crop)
    phase_zf = torch.angle(zf_complex_crop)

    # 3. Ground truth target reference
    if "target" in sample and sample["target"] is not None:
        target_tensor = sample["target"].to(device)
        if torch.is_complex(target_tensor):
            target_tensor = torch.abs(target_tensor)
        target_crop = center_crop(target_tensor, (320, 320))
    else:
        full_complex = kspace_to_image_torch(kspace)
        target_crop = center_crop(torch.abs(full_complex), (320, 320))

    # 4. Shared per-slice normalization
    zf_norm, target_norm, scale = normalize_slice_pair(zf_mag_crop, target_crop, percentile=99.0)
    y_raw_norm = y_raw / scale

    gt_np = target_norm.cpu().numpy().astype(np.float32)
    zf_np = zf_norm.cpu().numpy().astype(np.float32)

    data_range = float(gt_np.max() - gt_np.min() + 1e-12)

    results = {
        "filename": sample.get("filename", ""),
        "slice_idx": sample.get("slice_idx", 0),
        "gt": gt_np,
        "zf": zf_np,
        "zf_metrics": {
            "psnr": calculate_psnr(gt_np, zf_np, data_range=data_range),
            "ssim": calculate_ssim(gt_np, zf_np, data_range=data_range),
            "nmse": calculate_nmse(gt_np, zf_np),
        },
        "unet": None,
        "unet_metrics": None,
        "diffusion": None,
        "diffusion_metrics": None,
        "pinn": None,
        "pinn_metrics": None,
    }

    x_input = zf_norm.unsqueeze(0).unsqueeze(0).float().to(device)
    phase_tensor = phase_zf.unsqueeze(0).unsqueeze(0).float().to(device)
    mask_recon = get_recon_conditioning_mask(mask_raw, recon_size=(320, 320)).to(device)

    # 5. U-Net Evaluation
    if unet_model is not None:
        with torch.no_grad():
            unet_out = unet_model(x_input)
        unet_np = unet_out.squeeze().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
        results["unet"] = unet_np
        results["unet_metrics"] = {
            "psnr": calculate_psnr(gt_np, unet_np, data_range=data_range),
            "ssim": calculate_ssim(gt_np, unet_np, data_range=data_range),
            "nmse": calculate_nmse(gt_np, unet_np),
        }

    # 6. Diffusion-Only Evaluation (with measurement guidance)
    if diffusion_model is not None and diffusion_schedule is not None:
        with torch.no_grad():
            diff_out = diffusion_schedule.sample(
                model=diffusion_model,
                condition=x_input,
                mask=mask_recon,
                y_raw=y_raw_norm,
                mask_raw=mask_raw,
                phase_zf=phase_tensor,
                measurement_guided=True,
                device=device,
            )
        diff_np = diff_out.squeeze().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
        results["diffusion"] = diff_np
        results["diffusion_metrics"] = {
            "psnr": calculate_psnr(gt_np, diff_np, data_range=data_range),
            "ssim": calculate_ssim(gt_np, diff_np, data_range=data_range),
            "nmse": calculate_nmse(gt_np, diff_np),
        }

    # 7. Proposed Diffusion + PINN Evaluation
    if diffusion_pinn_model is not None:
        with torch.no_grad():
            coarse, refined = diffusion_pinn_model.sample(
                condition=x_input,
                mask=mask_recon,
                y_measured=y_raw_norm,
                mask_raw=mask_raw,
                phase_zf=phase_tensor,
                device=device,
            )
        pinn_np = refined.squeeze().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
        results["pinn"] = pinn_np
        results["pinn_metrics"] = {
            "psnr": calculate_psnr(gt_np, pinn_np, data_range=data_range),
            "ssim": calculate_ssim(gt_np, pinn_np, data_range=data_range),
            "nmse": calculate_nmse(gt_np, pinn_np),
        }
        if results["diffusion"] is None:
            coarse_np = coarse.squeeze().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            results["diffusion"] = coarse_np
            results["diffusion_metrics"] = {
                "psnr": calculate_psnr(gt_np, coarse_np, data_range=data_range),
                "ssim": calculate_ssim(gt_np, coarse_np, data_range=data_range),
                "nmse": calculate_nmse(gt_np, coarse_np),
            }

    return results


def plot_comparison_figure(
    results: dict,
    accel: int,
    save_path: str,
):
    """Saves a multi-panel visual comparison showing reconstructions and error maps."""
    methods = [("Ground Truth", results["gt"], None),
               ("Zero-Filled", results["zf"], results["zf_metrics"])]

    if results["unet"] is not None:
        methods.append(("U-Net Baseline", results["unet"], results["unet_metrics"]))
    if results["diffusion"] is not None:
        methods.append(("Diffusion Only", results["diffusion"], results["diffusion_metrics"]))
    if results["pinn"] is not None:
        methods.append(("Diffusion + PINN (Proposed)", results["pinn"], results["pinn_metrics"]))

    n_cols = len(methods)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    gt = results["gt"]

    for col, (title, img, m) in enumerate(methods):
        ax_img = axes[0, col] if axes.ndim == 2 else axes[col]
        ax_img.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        subtitle = title
        if m is not None:
            subtitle += f"\nPSNR: {m['psnr']:.2f} dB | SSIM: {m['ssim']:.4f}"
        ax_img.set_title(subtitle, fontsize=10)
        ax_img.axis("off")

        ax_err = axes[1, col] if axes.ndim == 2 else None
        if ax_err is not None:
            if title == "Ground Truth":
                ax_err.axis("off")
            else:
                err = np.abs(img - gt)
                ax_err.imshow(err, cmap="hot", vmin=0.0, vmax=0.3)
                ax_err.set_title(f"Error (NMSE: {m['nmse']:.4f})", fontsize=9)
                ax_err.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved benchmark figure: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion + PINN Architecture vs Baselines")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--accel", type=int, default=4, choices=[4, 8], help="Acceleration factor")
    parser.add_argument("--mask_type", type=str, default="equispaced", choices=["equispaced", "random"])
    parser.add_argument("--pinn_checkpoint", type=str, default=None, help="Path to Diffusion+PINN checkpoint")
    parser.add_argument("--unet_checkpoint", type=str, default=None, help="Path to U-Net checkpoint")
    parser.add_argument("--diffusion_checkpoint", type=str, default=None, help="Path to standalone Diffusion checkpoint")
    parser.add_argument("--subset_files", type=int, default=10, help="Number of validation files")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of slices to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic evaluation")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation directory")
    parser.add_argument("--device", type=str, default=None, help="Device")
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = load_yaml(args.config)
    val_dir = args.val_dir or cfg.get("data", {}).get("val_dir")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results_dir = Path("results")
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    accel = args.accel
    center_frac = 0.08 if accel == 4 else 0.04
    mask_func = create_mask_func(
        mask_type=args.mask_type,
        center_fraction=center_frac,
        acceleration=accel,
        seed=seed,
    )

    # 1. Load U-Net if provided
    unet_model = None
    if args.unet_checkpoint and Path(args.unet_checkpoint).exists():
        unet_model = UNet(in_channels=1, out_channels=1).to(device)
        ckpt = torch.load(args.unet_checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        unet_model.load_state_dict(state_dict)
        unet_model.eval()
        print(f"[MODEL] Loaded U-Net from: {args.unet_checkpoint}")

    # 2. Load Diffusion if provided
    diffusion_model = None
    diffusion_schedule = None
    diff_ckpt_path = args.diffusion_checkpoint
    if diff_ckpt_path is None:
        candidate = Path(f"checkpoints/diffusion_accel{accel}_best.pt")
        if candidate.exists():
            diff_ckpt_path = str(candidate)

    if diff_ckpt_path and Path(diff_ckpt_path).exists():
        diffusion_model = ConditionalDiffusionUNet(in_channels=3, base_channels=32, time_dim=128).to(device)
        ckpt = torch.load(diff_ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        diffusion_model.load_compatible_state_dict(state_dict)
        diffusion_model.eval()
        diffusion_schedule = DiffusionSchedule(timesteps=100, device=device)
        print(f"[MODEL] Loaded Diffusion from: {diff_ckpt_path}")

    # 3. Load Diffusion+PINN
    diffusion_pinn_model = None
    pinn_ckpt_path = args.pinn_checkpoint
    if pinn_ckpt_path is None:
        candidate = Path(f"checkpoints/diffusion_pinn_full_accel{accel}_best.pt")
        if candidate.exists():
            pinn_ckpt_path = str(candidate)

    if pinn_ckpt_path and Path(pinn_ckpt_path).exists():
        diffusion_pinn_model = DiffusionPINN(
            diffusion_base_channels=32,
            diffusion_time_dim=128,
            pinn_base_channels=32,
            pinn_num_blocks=4,
            timesteps=100,
            enforce_dc=True,
        ).to(device)
        ckpt = torch.load(pinn_ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        diffusion_pinn_model.load_state_dict(state_dict)
        diffusion_pinn_model.eval()
        print(f"[MODEL] Loaded Diffusion+PINN from: {pinn_ckpt_path}")

    if not val_dir or not Path(val_dir).exists():
        smoke_val = Path("data/fastmri/smoke/val")
        if smoke_val.exists() and len(list(smoke_val.glob("*.h5"))) > 0:
            print(f"[EVAL] Default validation directory '{val_dir}' not found.")
            print(f"[EVAL] Using smoke-test validation directory: {smoke_val}")
            val_dir = str(smoke_val)
        else:
            print(f"[WARN] Validation directory '{val_dir}' not found. Using realistic fastMRI knee phantom simulation.")
            H_raw, W_raw = 640, 368
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, H_raw),
                torch.linspace(-1, 1, W_raw),
                indexing="ij",
            )
            sim_img = torch.zeros(H_raw, W_raw, dtype=torch.float32)
            tissue = (x ** 2 / 0.55 ** 2 + y ** 2 / 0.75 ** 2) <= 1.0
            sim_img[tissue] = 0.65
            bone = (x ** 2 / 0.25 ** 2 + (y - 0.2) ** 2 / 0.25 ** 2) <= 1.0
            sim_img[bone] = 0.95
            phase = 0.5 * torch.sin(3 * x) + 0.2 * torch.cos(2 * y)
            sim_complex = sim_img.to(torch.complex64) * torch.exp(1j * phase)
            kspace_t = image_to_kspace_torch(sim_complex)
            target_t = center_crop(sim_img, (320, 320))
            val_dataset = [{
                "kspace": kspace_t,
                "target": target_t,
                "filename": "simulated_knee_sample",
                "slice_idx": 0,
            }]
    if 'val_dataset' not in locals():
        val_dataset = FastMRIDataset(val_dir, max_files=args.subset_files)
    total_eval = len(val_dataset) if args.max_samples is None else min(len(val_dataset), args.max_samples)
    print(f"[BENCHMARK] Evaluating {total_eval} validation slices at {accel}x acceleration...")

    all_results = []
    first_slice_res = None

    for idx in range(total_eval):
        sample = val_dataset[idx]
        res = evaluate_slice_all_methods(
            sample=sample,
            mask_func=mask_func,
            seed=seed,
            device=device,
            unet_model=unet_model,
            diffusion_model=diffusion_model,
            diffusion_schedule=diffusion_schedule,
            diffusion_pinn_model=diffusion_pinn_model,
        )
        if first_slice_res is None:
            first_slice_res = res

        row = {
            "filename": res["filename"],
            "slice_idx": res["slice_idx"],
            "zf_psnr": res["zf_metrics"]["psnr"],
            "zf_ssim": res["zf_metrics"]["ssim"],
            "zf_nmse": res["zf_metrics"]["nmse"],
        }
        if res["unet_metrics"]:
            row.update({
                "unet_psnr": res["unet_metrics"]["psnr"],
                "unet_ssim": res["unet_metrics"]["ssim"],
                "unet_nmse": res["unet_metrics"]["nmse"],
            })
        if res["diffusion_metrics"]:
            row.update({
                "diff_psnr": res["diffusion_metrics"]["psnr"],
                "diff_ssim": res["diffusion_metrics"]["ssim"],
                "diff_nmse": res["diffusion_metrics"]["nmse"],
            })
        if res["pinn_metrics"]:
            row.update({
                "pinn_psnr": res["pinn_metrics"]["psnr"],
                "pinn_ssim": res["pinn_metrics"]["ssim"],
                "pinn_nmse": res["pinn_metrics"]["nmse"],
            })
        all_results.append(row)

    df = pd.DataFrame(all_results)
    csv_path = results_dir / f"tables/diffusion_pinn_benchmark_accel{accel}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV metrics table: {csv_path}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY ({len(df)} slices evaluated at {accel}x acceleration)")
    print("=" * 70)
    print(f"Zero-Filled Baseline : PSNR: {df['zf_psnr'].mean():.2f} ± {df['zf_psnr'].std():.2f} dB | SSIM: {df['zf_ssim'].mean():.4f} ± {df['zf_ssim'].std():.4f} | NMSE: {df['zf_nmse'].mean():.4f}")
    if "unet_psnr" in df:
        print(f"U-Net Baseline       : PSNR: {df['unet_psnr'].mean():.2f} ± {df['unet_psnr'].std():.2f} dB | SSIM: {df['unet_ssim'].mean():.4f} ± {df['unet_ssim'].std():.4f} | NMSE: {df['unet_nmse'].mean():.4f}")
    if "diff_psnr" in df:
        print(f"Conditional Diffusion: PSNR: {df['diff_psnr'].mean():.2f} ± {df['diff_psnr'].std():.2f} dB | SSIM: {df['diff_ssim'].mean():.4f} ± {df['diff_ssim'].std():.4f} | NMSE: {df['diff_nmse'].mean():.4f}")
    if "pinn_psnr" in df:
        print(f"Diffusion + PINN     : PSNR: {df['pinn_psnr'].mean():.2f} ± {df['pinn_psnr'].std():.2f} dB | SSIM: {df['pinn_ssim'].mean():.4f} ± {df['pinn_ssim'].std():.4f} | NMSE: {df['pinn_nmse'].mean():.4f}")
    print("=" * 70)

    summary = {
        "accel": accel,
        "num_slices": len(df),
        "zero_filled": {
            "psnr_mean": float(df["zf_psnr"].mean()),
            "psnr_std": float(df["zf_psnr"].std()),
            "ssim_mean": float(df["zf_ssim"].mean()),
            "ssim_std": float(df["zf_ssim"].std()),
            "nmse_mean": float(df["zf_nmse"].mean()),
        },
    }
    if "diff_psnr" in df:
        summary["diffusion"] = {
            "psnr_mean": float(df["diff_psnr"].mean()),
            "psnr_std": float(df["diff_psnr"].std()),
            "ssim_mean": float(df["diff_ssim"].mean()),
            "ssim_std": float(df["diff_ssim"].std()),
            "nmse_mean": float(df["diff_nmse"].mean()),
        }
    if "pinn_psnr" in df:
        summary["pinn"] = {
            "psnr_mean": float(df["pinn_psnr"].mean()),
            "psnr_std": float(df["pinn_psnr"].std()),
            "ssim_mean": float(df["pinn_ssim"].mean()),
            "ssim_std": float(df["pinn_ssim"].std()),
            "nmse_mean": float(df["pinn_nmse"].mean()),
        }

    summary_path = results_dir / f"logs/diffusion_pinn_benchmark_accel{accel}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save visual comparison for first slice
    if first_slice_res is not None:
        fig_path = str(results_dir / f"figures/diffusion_pinn_accel{accel}_comparison.png")
        plot_comparison_figure(first_slice_res, accel=accel, save_path=fig_path)


if __name__ == "__main__":
    main()
