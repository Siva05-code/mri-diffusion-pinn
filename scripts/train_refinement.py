"""Stage-2 entry point: train the physics-guided refinement U-Net."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_refinement import run_refinement_training


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--diffusion_checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    run_refinement_training(
        args.config, args.train_dir, args.val_dir,
        args.diffusion_checkpoint, device,
        output_dir=args.output_dir, smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
