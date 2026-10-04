"""Stage-1 entry point: train the conditional diffusion reconstructor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_diffusion import run_training


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    run_training(
        args.config,
        args.train_dir,
        args.val_dir,
        device,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
