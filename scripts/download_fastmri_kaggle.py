"""
FastMRI Dataset Downloader via Kaggle API.

Downloads official NYU FastMRI single-coil knee .h5 files using the local
Kaggle CLI / credentials into:
  - data/fastmri/knee_singlecoil_train/
  - data/fastmri/knee_singlecoil_val/
"""

import subprocess
import sys
from pathlib import Path

TRAIN_DATASET = "arafatshovon/fastmri-train-knee-2"
VAL_DATASET = "moshiurrahman00/nyu-fastmri-knee-singlecoil-val"

DEFAULT_TRAIN_FILES = [
    "file1001239.h5",
    "file1001242.h5",
    "file1001244.h5",
    "file1001245.h5",
    "file1001246.h5",
]

DEFAULT_VAL_FILES = [
    "file1000000.h5",
    "file1000007.h5",
    "file1000017.h5",
]


def download_file(dataset: str, filename: str, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / filename
    if target_file.exists() and target_file.stat().st_size > 10 * 1024 * 1024:
        print(f"  [Skipped - exists] {filename} ({target_file.stat().st_size / (1024*1024):.1f} MB)")
        return True

    print(f"  [Downloading] {filename} from {dataset}...")
    cmd = [
        "kaggle", "datasets", "download",
        "-f", filename,
        dataset,
        "-p", str(target_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Retry with direct python kaggle path if kaggle not in standard PATH
        cmd[0] = "/Library/Frameworks/Python.framework/Versions/3.13/bin/kaggle"
        res = subprocess.run(cmd, capture_output=True, text=True)

    zip_file = target_dir / f"{filename}.zip"
    if res.returncode == 0:
        if zip_file.exists():
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(target_dir)
            zip_file.unlink()
        print(f"  [Downloaded & Extracted OK] {filename}")
        return True
    else:
        print(f"  [Error downloading {filename}]: {res.stderr.strip()}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download real FastMRI knee files via Kaggle API.")
    parser.add_argument("--num_train", type=int, default=5, help="Number of training H5 files to download")
    parser.add_argument("--num_val", type=int, default=3, help="Number of validation H5 files to download")
    args = parser.parse_args()

    train_dir = Path("data/fastmri/knee_singlecoil_train")
    val_dir = Path("data/fastmri/knee_singlecoil_val")

    print("=" * 60)
    print("DOWNLOADING REAL FASTMRI KNEE DATASET VIA KAGGLE API")
    print("=" * 60)
    print(f"Train directory: {train_dir}")
    print(f"Val directory  : {val_dir}")
    print("=" * 60)

    print("\n[1/2] Downloading Training Files...")
    train_files = DEFAULT_TRAIN_FILES[:args.num_train]
    for f in train_files:
        download_file(TRAIN_DATASET, f, train_dir)

    print("\n[2/2] Downloading Validation Files...")
    val_files = DEFAULT_VAL_FILES[:args.num_val]
    for f in val_files:
        download_file(VAL_DATASET, f, val_dir)

    print("\n" + "=" * 60)
    total_train = len(list(train_dir.glob("*.h5")))
    total_val = len(list(val_dir.glob("*.h5")))
    print(f"Setup complete! Real fastMRI files on disk:")
    print(f"  - Train: {total_train} files in {train_dir}")
    print(f"  - Val  : {total_val} files in {val_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

