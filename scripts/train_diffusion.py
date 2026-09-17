#!/usr/bin/env python3
"""
CLI Training Script for Conditional Diffusion MRI Reconstruction Baseline.

Usage:
    # 1. Pipeline Smoke Test (uses validation subset for code verification):
    python scripts/train_diffusion.py --config configs/base.yaml --accel 4 --epochs 1 --subset_files 2 --smoke_test

    # 2. Real Training on Official Training Set:
    python scripts/train_diffusion.py --config configs/base.yaml --accel 4 --epochs 5
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

from src.training.train_diffusion import train_diffusion


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(
        description="Train conditional diffusion model for accelerated MRI reconstruction."
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base configuration YAML.")
    parser.add_argument("--accel", type=int, default=4, choices=[4, 8], help="Acceleration factor (4 or 8).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1).")
    parser.add_argument("--timesteps", type=int, default=None, help="Diffusion timesteps (default: 100).")
    parser.add_argument("--subset_files", type=int, default=None, help="Limit number of H5 files used.")
    parser.add_argument("--lambda_image", type=float, default=None, help="Weight for image L1 loss.")
    parser.add_argument("--lambda_kspace", type=float, default=None, help="Weight for kspace loss.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Enable smoke-test mode (allows pipeline verification on validation subset).",
    )
    parser.add_argument("--train_dir", type=str, default=None, help="Path to training data directory.")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation data directory.")
    parser.add_argument("--device", type=str, default=None, help="Execution device ('cuda' or 'cpu').")
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Maximum number of validation batches. Use -1 for full validation.",
    )
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
        cfg.setdefault("physics", {}).update(accel_cfg.get("physics", {}))

    # 3. Resolve hyperparameters
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    train_cfg = cfg.get("training", {})
    diff_cfg = cfg.get("diffusion", {})
    phys_cfg = cfg.get("physics", {})

    train_dir = args.train_dir or data_cfg.get("train_dir")
    val_dir = args.val_dir or data_cfg.get("val_dir")

    # Sanity checks and validation-fallback gating
    if not args.smoke_test:
        if not train_dir or not Path(train_dir).exists():
            raise FileNotFoundError(
                f"Official training directory '{train_dir}' does not exist.\n"
                f"Real scientific training requires the official fastMRI training dataset.\n"
                f"Training on the validation set is strictly prohibited for real experiments.\n"
                f"If you only intend to verify code and execution pipelines, run with --smoke_test."
            )
        if val_dir and Path(train_dir).resolve() == Path(val_dir).resolve():
            raise ValueError(
                f"FATAL: Training directory ('{train_dir}') is identical to validation directory ('{val_dir}').\n"
                f"Training on the validation set is prohibited for real experiments.\n"
                f"To verify the pipeline using validation data, explicitly pass --smoke_test."
            )
    else:
        # Smoke test mode: allow using validation directory only for pipeline verification
        # Smoke test mode: allow using validation directory or local simulation for pipeline verification
        if not train_dir or not Path(train_dir).exists():
            if val_dir and Path(val_dir).exists():
                print(f"[SMOKE TEST] Training directory '{train_dir}' not found.")
                print(f"[SMOKE TEST] Using validation subset '{val_dir}' strictly for pipeline verification.")
                train_dir = val_dir
            else:
                print("[SMOKE TEST] FastMRI dataset directories not found locally.")
                print("[SMOKE TEST] Generating realistic simulated fastMRI knee dataset for pipeline verification.")
                from src.physics.fft import image_to_kspace_torch
                from src.data.preprocessing import center_crop
                import h5py
                import torch
                import numpy as np

                smoke_root = Path("data/fastmri/smoke")
                smoke_train = smoke_root / "train"
                smoke_val = smoke_root / "val"
                smoke_train.mkdir(parents=True, exist_ok=True)
                smoke_val.mkdir(parents=True, exist_ok=True)

                H, W = 640, 368
                y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
                knee = (x**2 / 0.6**2 + y**2 / 0.8**2) < 0.8
                bone = ((x + 0.1)**2 / 0.2**2 + (y + 0.1)**2 / 0.3**2) < 0.5
                sim_img = torch.zeros(H, W, dtype=torch.float32)
                sim_img[knee] = 0.6 + 0.3 * torch.cos(8 * x[knee])
                sim_img[bone] = 0.95
                phase = 0.5 * torch.sin(3 * x) + 0.2 * torch.cos(2 * y)
                sim_complex = sim_img.to(torch.complex64) * torch.exp(1j * phase)
                kspace_np = image_to_kspace_torch(sim_complex).numpy()
                target_np = center_crop(sim_img, (320, 320)).numpy()

                kspace_batch = np.stack([kspace_np, kspace_np], axis=0)
                target_batch = np.stack([target_np, target_np], axis=0)

                for folder in [smoke_train, smoke_val]:
                    fname = folder / "sim_knee_sample.h5"
                    if not fname.exists():
                        with h5py.File(fname, "w") as hf:
                            hf.create_dataset("kspace", data=kspace_batch)
                            hf.create_dataset("reconstruction_rss", data=target_batch)
                train_dir = str(smoke_train)
                val_dir = str(smoke_val)

    epochs = args.epochs or train_cfg.get("epochs", 5)
    learning_rate = args.lr or train_cfg.get("learning_rate", 1e-4)
    batch_size = args.batch_size or train_cfg.get("batch_size", 1)
    acceleration = args.accel or exp_cfg.get("acceleration", 4)
    timesteps = args.timesteps or diff_cfg.get("timesteps", 100)
    device = args.device or "cuda"

    subset_files = args.subset_files
    if subset_files is None and args.smoke_test:
        subset_files = 2

    # 4. Call existing implementation from src/training/train_diffusion.py
    lambda_image = args.lambda_image if args.lambda_image is not None else phys_cfg.get("lambda_image", 1.0)
    lambda_kspace = args.lambda_kspace if args.lambda_kspace is not None else phys_cfg.get("lambda_kspace", 0.1)
    seed = args.seed if args.seed is not None else exp_cfg.get("seed", 42)
    max_val_batches = args.max_val_batches

    # 4. Call implementation from src/training/train_diffusion.py
    train_diffusion(
        train_dir=train_dir,
        val_dir=val_dir,
        acceleration=acceleration,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        timesteps=timesteps,
        subset_files=subset_files,
        lambda_image=lambda_image,
        lambda_kspace=lambda_kspace,
        seed=seed,
        max_val_batches=max_val_batches,
        device=device,
    )


if __name__ == "__main__":
    main()

