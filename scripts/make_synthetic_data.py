"""
Generate synthetic single-coil "MRI" volumes in fastMRI HDF5 layout.

Purpose: let you verify that every stage of the pipeline runs, trains and
improves before you have fastMRI access, and give the unit tests something
deterministic to work on. These are ellipse phantoms with texture and noise --
useful for checking the machinery, not for reporting results.

Usage:
    python scripts/make_synthetic_data.py --out data/synthetic --train 6 --val 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _ellipse(grid_y, grid_x, cy, cx, ry, rx, angle):
    ct, st = np.cos(angle), np.sin(angle)
    y = (grid_y - cy) * ct + (grid_x - cx) * st
    x = -(grid_y - cy) * st + (grid_x - cx) * ct
    return (y / ry) ** 2 + (x / rx) ** 2 <= 1.0


def make_phantom(size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size), dtype=np.float32)

    # Outer body
    body = _ellipse(yy, xx, size / 2, size / 2, size * 0.42, size * 0.34, 0.0)
    img[body] = 0.45

    # Internal structures
    for _ in range(rng.integers(5, 10)):
        img[
            _ellipse(
                yy, xx,
                rng.uniform(0.3, 0.7) * size,
                rng.uniform(0.3, 0.7) * size,
                rng.uniform(0.04, 0.16) * size,
                rng.uniform(0.04, 0.16) * size,
                rng.uniform(0, np.pi),
            )
        ] = rng.uniform(0.15, 1.0)

    # Fine texture so the high frequencies are not trivially zero
    freq = rng.uniform(0.15, 0.4)
    texture = 0.06 * np.sin(freq * xx + rng.uniform(0, 6)) * np.cos(freq * yy)
    img = np.clip(img + texture * body, 0.0, 1.0)

    # Thin bright lines: structure that only lives in the outer k-space
    for _ in range(3):
        r = int(rng.integers(int(0.25 * size), int(0.75 * size)))
        img[r:r + 1, int(0.25 * size):int(0.75 * size)] = 0.95

    img = img + rng.normal(0, 0.005, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--train", type=int, default=6, help="number of train volumes")
    p.add_argument("--val", type=int, default=2, help="number of val volumes")
    p.add_argument("--slices", type=int, default=12)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.out)
    rng = np.random.default_rng(args.seed)

    for split, count in (("train", args.train), ("val", args.val)):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        for v in range(count):
            recon = np.stack(
                [make_phantom(args.size, rng) for _ in range(args.slices)]
            ).astype(np.float32)
            kspace = np.stack([fft2c(s) for s in recon]).astype(np.complex64)

            with h5py.File(folder / f"synthetic_{split}_{v:03d}.h5", "w") as f:
                f.create_dataset("kspace", data=kspace)
                f.create_dataset("reconstruction_esc", data=recon)
                f.attrs["acquisition"] = "SYNTHETIC"
        print(f"{split}: {count} volumes x {args.slices} slices -> {folder}")

    print("\nThese are phantoms, not real anatomy. Use them to validate the "
          "pipeline, then switch to fastMRI for any reported number.")


if __name__ == "__main__":
    main()
