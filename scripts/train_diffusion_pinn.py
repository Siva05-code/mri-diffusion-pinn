#!/usr/bin/env python3
"""
CLI Training Script for Proposed Physics-Informed Diffusion-PINN Architecture.

Usage:
    # 1. Smoke test on validation subset (for code verification):
    python3 scripts/train_diffusion_pinn.py --config configs/base.yaml --accel 4 --epochs 1 --subset_files 2 --smoke_test

    # 2. Real Kaggle GPU Training (4x acceleration):
    python3 scripts/train_diffusion_pinn.py --config configs/base.yaml --accel 4 --epochs 5 --subset_files 20

    # 3. Real Kaggle GPU Training (8x acceleration):
    python3 scripts/train_diffusion_pinn.py --config configs/base.yaml --accel 8 --epochs 5 --subset_files 20

    # 4. Run Ablation (e.g. Diffusion only):
    python3 scripts/train_diffusion_pinn.py --config configs/base.yaml --accel 4 --ablation diffusion_only
"""

import sys
import os
import argparse
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_diffusion_pinn import train_diffusion_pinn


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(
        description="Train Physics-Informed Diffusion-PINN for Accelerated MRI Reconstruction."
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base configuration YAML.")
    parser.add_argument("--accel", type=int, default=4, choices=[4, 8], help="Acceleration factor (4 or 8).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--subset_files", type=int, default=None, help="Limit number of H5 files used.")
    parser.add_argument("--timesteps", type=int, default=100, help="Diffusion timesteps.")
    parser.add_argument("--lambda_image", type=float, default=None, help="Weight for image L1 loss.")
    parser.add_argument("--lambda_kspace", type=float, default=None, help="Weight for kspace loss.")
    parser.add_argument("--lambda_physics", type=float, default=None, help="Weight for physics loss.")
    parser.add_argument("--lambda_gradient", type=float, default=None, help="Weight for gradient TV loss.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "diffusion_only", "diffusion_kspace", "diffusion_physics", "diffusion_pinn_no_physics"],
        help="Ablation experiment mode.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Enable smoke-test mode (allows pipeline verification on validation subset).",
    )
    parser.add_argument("--train_dir", type=str, default=None, help="Path to training data directory.")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation data directory.")
    parser.add_argument("--device", type=str, default=None, help="Execution device ('cuda' or 'cpu').")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    accel_yaml = Path(f"configs/accel{args.accel}.yaml")
    if accel_yaml.exists():
        accel_cfg = load_yaml(str(accel_yaml))
        cfg.setdefault("experiment", {}).update(accel_cfg.get("experiment", {}))
        cfg.setdefault("physics", {}).update(accel_cfg.get("physics", {}))

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    phys_cfg = cfg.get("physics", {})
    diff_cfg = cfg.get("diffusion", {})

    train_dir = args.train_dir or data_cfg.get("train_dir")
    val_dir = args.val_dir or data_cfg.get("val_dir")

    if not args.smoke_test:
        if not train_dir or not Path(train_dir).exists():
            raise FileNotFoundError(
                f"Official training directory '{train_dir}' does not exist.\n"
                f"Real training requires official fastMRI files.\n"
                f"For quick code verification, run with --smoke_test."
            )
    else:
        if not train_dir or not Path(train_dir).exists():
            if val_dir and Path(val_dir).exists():
                print(f"[SMOKE TEST] Using validation subset '{val_dir}' for pipeline verification.")
                train_dir = val_dir

    epochs = args.epochs or train_cfg.get("epochs", 5)
    learning_rate = args.lr or train_cfg.get("learning_rate", 1e-4)
    batch_size = args.batch_size or train_cfg.get("batch_size", 1)
    timesteps = args.timesteps or diff_cfg.get("timesteps", 100)
    device = args.device or ("cuda" if Path("/dev/nvidia0").exists() else "cpu")

    train_diffusion_pinn(
        train_dir=train_dir,
        val_dir=val_dir,
        acceleration=args.accel,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        timesteps=timesteps,
        subset_files=args.subset_files,
        lambda_image=args.lambda_image if args.lambda_image is not None else phys_cfg.get("lambda_image", 1.0),
        lambda_kspace=args.lambda_kspace if args.lambda_kspace is not None else phys_cfg.get("lambda_kspace", 0.1),
        lambda_physics=args.lambda_physics if args.lambda_physics is not None else phys_cfg.get("lambda_physics", 0.1),
        lambda_gradient=args.lambda_gradient if args.lambda_gradient is not None else phys_cfg.get("lambda_gradient", 0.1),
        ablation=args.ablation,
        seed=args.seed if args.seed is not None else exp_cfg.get("seed", 42),
        device=device,
    )


if __name__ == "__main__":
    main()

