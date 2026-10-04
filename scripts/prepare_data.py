"""
Convert fastMRI HDF5 volumes into a single float32 .npy array of 320x320 targets.

Why bother: the raw path runs a 640x372 inverse FFT for every slice, every
epoch. Caching turns that into a memmap read and typically makes a Colab epoch
several times faster, which matters far more than it sounds when you have a
12-hour session limit.

Usage:
    python scripts/prepare_data.py --src data/fastmri/train --out data/cache/train.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import center_crop, kspace_to_target, normalize_magnitude


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder with fastMRI .h5 files")
    p.add_argument("--out", required=True, help="output .npy path")
    p.add_argument("--image_size", type=int, default=320)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--slice_low", type=float, default=0.15)
    p.add_argument("--slice_high", type=float, default=0.95)
    args = p.parse_args()

    files = sorted(list(Path(args.src).rglob("*.h5")) + list(Path(args.src).rglob("*.hdf5")))
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise SystemExit(f"No .h5 files under {args.src}")

    slices = []
    for path in tqdm(files, desc="volumes"):
        with h5py.File(path, "r") as f:
            use_recon = "reconstruction_esc" in f
            key = "reconstruction_esc" if use_recon else "kspace"
            if key not in f:
                continue
            n = int(f[key].shape[0])
            lo, hi = int(n * args.slice_low), int(n * args.slice_high)
            lo, hi = max(lo, 0), max(min(hi, n), 1)
            for s in range(lo, hi):
                if use_recon:
                    img = np.asarray(f["reconstruction_esc"][s], dtype=np.float32)
                    img = center_crop(img, args.image_size)
                    target, _ = normalize_magnitude(img)
                else:
                    target, _ = kspace_to_target(np.asarray(f["kspace"][s]), args.image_size)
                slices.append(target)

    if not slices:
        raise SystemExit("No usable slices found.")

    arr = np.stack(slices).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    print(f"Saved {arr.shape} -> {out}  ({arr.nbytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
