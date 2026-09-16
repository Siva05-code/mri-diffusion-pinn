#!/usr/bin/env python3
"""
Benchmark Evaluation Script: Zero-Filled vs U-Net vs Diffusion vs Proposed Diffusion+PINN.

Evaluates on the exact same fastMRI validation slices and reproducible sampling masks:
1. Zero-Filled (baseline)
2. U-Net (baseline)
3. Conditional Diffusion (pre-reconstruction baseline)
4. Diffusion + PINN (proposed architecture)

Calculates:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- NMSE (Normalized Mean Squared Error)

Saves results to:
- results/tables/diffusion_pinn_benchmark_metrics.csv
- results/logs/diffusion_pinn_evaluation_summary.json
- results/figures/diffusion_pinn_accel{accel}_comparison.png
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
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.unet import UNet
from src.models.diffusion import ConditionalDiffusionUNet
from src.models.diffusion_schedule import DiffusionSchedule
from src.models.diffusion_pinn import DiffusionPINN
from src.data.dataset import FastMRIDataset
from src.data.undersampling import RandomMaskFunc
from src.data.preprocessing import complex_abs, align_and_normalize_slice, center_crop
from src.physics.fft import kspace_to_image_torch, image_to_kspace_torch
from src.evaluation.metrics import calculate_psnr, calculate_ssim, calculate_nmse
from src.visualization.error_maps import compute_absolute_error_map


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def evaluate_slice_all_methods(
    sample: dict,
    mask_func: RandomMaskFunc,
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
    mask = mask_func(kspace.shape, seed=slice_seed).to(device)

    # 1. Undersample k-space
    y_undersampled = mask * kspace

    # 2. Zero-Filled reconstruction via centered IFFT
    zf_complex = kspace_to_image_torch(y_undersampled)
    zf_mag = torch.abs(zf_complex)

    # 3. Reference ground truth
    if "target" in sample and sample["target"] is not None:
        target_mag = sample["target"].to(device)
        if torch.is_complex(target_mag):
            target_mag = torch.abs(target_mag)
    else:
        gt_complex = kspace_to_image_torch(kspace)
        target_mag = torch.abs(gt_complex)

    # 4. Spatial alignment & [0, 1] normalization
    zf_norm, target_norm = align_and_normalize_slice(zf_mag, target_mag)

    gt_np = target_norm.cpu().numpy()
    zf_np = zf_norm.cpu().numpy()

    results = {
        "filename": sample.get("filename", ""),
        "slice_idx": sample.get("slice_idx", 0),
        "gt": gt_np,
        "zf": zf_np,
        "zf_metrics": {
            "psnr": calculate_psnr(gt_np, zf_np),
            "ssim": calculate_ssim(gt_np, zf_np),
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

    # 5. U-Net Evaluation
    if unet_model is not None:
        with torch.no_grad():
            unet_out = unet_model(x_input)
        unet_np = unet_out.squeeze().clamp(0.0, 1.0).cpu().numpy()
        results["unet"] = unet_np
        results["unet_metrics"] = {
            "psnr": calculate_psnr(gt_np, unet_np),
            "ssim": calculate_ssim(gt_np, unet_np),
            "nmse": calculate_nmse(gt_np, unet_np),
        }

    # 6. Diffusion-Only Evaluation
    if diffusion_model is not None and diffusion_schedule is not None:
        with torch.no_grad():
            diff_out = diffusion_schedule.sample(
                model=diffusion_model,
                condition=x_input,
                device=device,
            )
        diff_np = diff_out.squeeze().clamp(0.0, 1.0).cpu().numpy()
        results["diffusion"] = diff_np
        results["diffusion_metrics"] = {
            "psnr": calculate_psnr(gt_np, diff_np),
            "ssim": calculate_ssim(gt_np, diff_np),
            "nmse": calculate_nmse(gt_np, diff_np),
        }

    # 7. Proposed Diffusion + PINN Evaluation
    if diffusion_pinn_model is not None:
        with torch.no_grad():
            coarse, refined = diffusion_pinn_model.sample(
                condition=x_input,
                y_measured=y_undersampled,
                mask=mask,
                device=device,
            )
        pinn_np = refined.squeeze().clamp(0.0, 1.0).cpu().numpy()
        results["pinn"] = pinn_np
        results["pinn_metrics"] = {
            "psnr": calculate_psnr(gt_np, pinn_np),
            "ssim": calculate_ssim(gt_np, pinn_np),
            "nmse": calculate_nmse(gt_np, pinn_np),
        }
        # If standalone diffusion was not loaded, use coarse from diffusion_pinn
        if results["diffusion"] is None:
            coarse_np = coarse.squeeze().clamp(0.0, 1.0).cpu().numpy()
            results["diffusion"] = coarse_np
            results["diffusion_metrics"] = {
                "psnr": calculate_psnr(gt_np, coarse_np),
                "ssim": calculate_ssim(gt_np, coarse_np),
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
        # Row 0: Image
        ax_img = axes[0, col] if axes.ndim == 2 else axes[col]
        ax_img.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        subtitle = title
        if m is not None:
            subtitle += f"\nPSNR: {m['psnr']:.2f} dB | SSIM: {m['ssim']:.4f}"
        ax_img.set_title(subtitle, fontsize=10)
        ax_img.axis("off")

        # Row 1: Absolute Error Map
        ax_err = axes[1, col] if axes.ndim == 2 else None
        if ax_err is not None:
            if title == "Ground Truth":
                ax_err.axis("off")
            else:
                err = np.abs(img - gt)
                im_err = ax_err.imshow(err, cmap="hot", vmin=0.0, vmax=0.3)
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
    parser.add_argument("--pinn_checkpoint", type=str, default=None, help="Path to Diffusion+PINN checkpoint")
    parser.add_argument("--unet_checkpoint", type=str, default=None, help="Path to U-Net checkpoint")
    parser.add_argument("--diffusion_checkpoint", type=str, default=None, help="Path to standalone Diffusion checkpoint")
    parser.add_argument("--subset_files", type=int, default=5, help="Number of validation files")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of slices to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic evaluation")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation directory")
    parser.add_argument("--device", type=str, default=None, help="Device")
    args = parser.parse_args()

    # Set seeds for reproducible evaluation
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
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)

    accel = args.accel
    center_frac = 0.08 if accel == 4 else 0.04
    mask_func = RandomMaskFunc(center_fractions=[center_frac], accelerations=[accel], seed=seed)

    # 1. Load Proposed Diffusion + PINN
    diffusion_pinn_model = None
    pinn_ckpt_path = args.pinn_checkpoint or f"checkpoints/diffusion_pinn_accel{accel}_best.pt"
    if Path(pinn_ckpt_path).exists():
        diffusion_pinn_model = DiffusionPINN(
            diffusion_base_channels=32,
            diffusion_time_dim=128,
            pinn_base_channels=32,
            pinn_num_blocks=4,
            timesteps=100,
        ).to(device)
        ckpt = torch.load(pinn_ckpt_path, map_location=device)
        diffusion_pinn_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        diffusion_pinn_model.eval()
        print(f"Loaded Diffusion+PINN checkpoint from: {pinn_ckpt_path}")
    else:
        print(f"[INFO] No Diffusion+PINN checkpoint found at '{pinn_ckpt_path}'.")

    # 2. Load U-Net Baseline
    unet_model = None
    unet_ckpt_path = args.unet_checkpoint or f"checkpoints/unet_accel{accel}_best.pt"
    if Path(unet_ckpt_path).exists():
        unet_model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
        ckpt = torch.load(unet_ckpt_path, map_location=device)
        unet_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        unet_model.eval()
        print(f"Loaded U-Net checkpoint from: {unet_ckpt_path}")
    else:
        print(f"[INFO] No U-Net checkpoint found at '{unet_ckpt_path}'.")

    # 3. Load Standalone Diffusion
    diffusion_model = None
    diffusion_schedule = None
    diff_ckpt_path = args.diffusion_checkpoint or f"checkpoints/diffusion_accel{accel}_latest.pt"
    if Path(diff_ckpt_path).exists():
        diffusion_model = ConditionalDiffusionUNet(base_channels=32, time_dim=128).to(device)
        ckpt = torch.load(diff_ckpt_path, map_location=device)
        diffusion_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        diffusion_model.eval()
        diffusion_schedule = DiffusionSchedule(timesteps=100, device=device)
        print(f"Loaded Diffusion checkpoint from: {diff_ckpt_path}")
    else:
        print(f"[INFO] No standalone Diffusion checkpoint found at '{diff_ckpt_path}'.")

    # Load dataset or generate phantom sample
    if not val_dir or not Path(val_dir).exists() or len(list(Path(val_dir).glob("*.h5"))) == 0:
        print("[DATA] FastMRI validation directory not found locally.")
        print("[DATA] Generating realistic fastMRI knee phantom for pipeline benchmark verification.")
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
        samples = [{"kspace": kspace_t, "target": target_t, "filename": "simulated_knee.h5", "slice_idx": 0}]
    else:
        dataset = FastMRIDataset(val_dir, max_files=args.subset_files)
        max_slices = args.max_samples if args.max_samples is not None else len(dataset)
        samples = [dataset[i] for i in range(min(len(dataset), max_slices))]

    print(f"Evaluating {len(samples)} slices at {accel}x acceleration (seed={seed})...")

    records = []
    first_res = None

    for idx, sample in enumerate(samples):
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
        if first_res is None:
            first_res = res

        # Append records
        records.append({"method": "Zero-Filled", "acceleration": accel, "slice": idx, **res["zf_metrics"]})
        if res["unet_metrics"]:
            records.append({"method": "U-Net", "acceleration": accel, "slice": idx, **res["unet_metrics"]})
        if res["diffusion_metrics"]:
            records.append({"method": "Diffusion-Only", "acceleration": accel, "slice": idx, **res["diffusion_metrics"]})
        if res["pinn_metrics"]:
            records.append({"method": "Diffusion+PINN", "acceleration": accel, "slice": idx, **res["pinn_metrics"]})

    # Save DataFrame
    df = pd.DataFrame(records)
    csv_path = results_dir / "tables" / f"diffusion_pinn_benchmark_accel{accel}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved slice benchmark metrics to: {csv_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"BENCHMARK SUMMARY ({accel}x Acceleration, N={len(samples)} slices)")
    print("=" * 60)
    summary_data = []
    for method, group in df.groupby("method"):
        p_mean, p_std = group["psnr"].mean(), group["psnr"].std()
        s_mean, s_std = group["ssim"].mean(), group["ssim"].std()
        n_mean, n_std = group["nmse"].mean(), group["nmse"].std()
        print(f"[{method}]:")
        print(f"  PSNR : {p_mean:.2f} ± {p_std:.2f} dB")
        print(f"  SSIM : {s_mean:.4f} ± {s_std:.4f}")
        print(f"  NMSE : {n_mean:.4f} ± {n_std:.4f}")
        summary_data.append({
            "method": method,
            "acceleration": accel,
            "psnr_mean": round(float(p_mean), 3),
            "psnr_std": round(float(p_std), 3),
            "ssim_mean": round(float(s_mean), 4),
            "ssim_std": round(float(s_std), 4),
            "nmse_mean": round(float(n_mean), 5),
            "nmse_std": round(float(n_std), 5),
        })

    # Save figure if representative slice available
    if first_res is not None:
        fig_path = str(results_dir / "figures" / f"diffusion_pinn_accel{accel}_comparison.png")
        plot_comparison_figure(first_res, accel, fig_path)

    # Save summary json
    summary_json_path = results_dir / "logs" / f"diffusion_pinn_benchmark_accel{accel}_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()

