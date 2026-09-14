#!/usr/bin/env python3
"""
CLI Training Script for U-Net MRI Reconstruction Baseline.

Usage:
    # Quick smoke test on 2 files for 1 epoch:
    python3 scripts/train.py --config configs/base.yaml --accel 4 --epochs 1 --subset_files 2 --smoke_test

    # Standard training on 4x acceleration:
    python3 scripts/train.py --config configs/base.yaml --accel 4 --epochs 10

    # Standard training on 8x acceleration:
    python3 scripts/train.py --config configs/base.yaml --accel 8 --epochs 10
"""

import sys
import os
import argparse
from pathlib import Path
import yaml

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_unet import train_unet


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Train U-Net baseline for accelerated MRI reconstruction.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base configuration YAML.")
    parser.add_argument("--accel", type=int, default=4, choices=[4, 8], help="Acceleration factor (4 or 8).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1).")
    parser.add_argument("--subset_files", type=int, default=None, help="Limit number of H5 files used (for quick tests).")
    parser.add_argument("--smoke_test", action="store_true", help="Designate run as a smoke test.")
    parser.add_argument("--train_dir", type=str, default=None, help="Path to training data directory.")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation data directory.")
    parser.add_argument("--device", type=str, default=None, help="Execution device ('cuda' or 'cpu').")
    args = parser.parse_args()

    # 1. Load base configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Base configuration file not found at: {config_path}")
    cfg = load_yaml(str(config_path))

    # 2. Merge acceleration-specific config if available
    accel_yaml = Path(f"configs/accel{args.accel}.yaml")
    if accel_yaml.exists():
        accel_cfg = load_yaml(str(accel_yaml))
        cfg.setdefault("experiment", {}).update(accel_cfg.get("experiment", {}))

    # 3. Resolve hyperparameters with CLI overrides
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    train_cfg = cfg.get("training", {})
    paths_cfg = cfg.get("paths", {})

    train_dir = args.train_dir or data_cfg.get("train_dir")
    val_dir = args.val_dir or data_cfg.get("val_dir")

    # If train directory does not exist, but val_dir does (e.g. initial download stage),
    # allow running smoke test on the available validation subset.
    if not Path(train_dir).exists():
        if val_dir and Path(val_dir).exists():
            print(f"[Note] Training directory '{train_dir}' not found.")
            print(f"[Note] Falling back to validation directory '{val_dir}' for code verification / smoke testing.")
            train_dir = val_dir
            args.smoke_test = True
        else:
            raise FileNotFoundError(
                f"Neither training directory ('{train_dir}') nor validation directory ('{val_dir}') exists.\n"
                f"Please ensure official fastMRI data has been downloaded."
            )

    epochs = args.epochs or train_cfg.get("epochs", 5)
    learning_rate = args.lr or train_cfg.get("learning_rate", 1e-4)
    batch_size = args.batch_size or train_cfg.get("batch_size", 1)
    acceleration = args.accel or exp_cfg.get("acceleration", 4)
    center_fraction = 0.08 if acceleration == 4 else 0.04
    seed = exp_cfg.get("seed", 42)

    subset_files = args.subset_files
    if subset_files is None and args.smoke_test:
        subset_files = 2  # default small file count for smoke test

    checkpoints_dir = paths_cfg.get("checkpoints_dir", "checkpoints")
    results_dir = paths_cfg.get("results_dir", "results")
    logs_dir = os.path.join(results_dir, "logs")

    # 4. Launch training
    train_unet(
        data_dir=train_dir,
        val_data_dir=val_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        acceleration=acceleration,
        center_fraction=center_fraction,
        subset_files=subset_files,
        device_str=args.device,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        seed=seed,
        is_smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()

