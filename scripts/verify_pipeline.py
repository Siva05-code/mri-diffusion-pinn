"""
Official NYU FastMRI Data Pipeline & Physics Verification Script.

Verifies end-to-end functionality using ACTUAL official fastMRI HDF5 files:
1. Resolves dataset directory from configs/base.yaml (val_dir / dataset_root)
2. Validates official HDF5 keys and k-space shape
3. Loads a valid slice
4. Reconstructs ground-truth image via centered 2D IFFT
5. Applies 4x and 8x Cartesian undersampling masks
6. Performs zero-filled inverse FFT reconstruction
7. Calculates quantitative metrics (PSNR, SSIM, NMSE, k-space error)
8. Generates and saves 5-panel reconstruction comparison figures under results/figures/

NOTE: Does NOT use or generate synthetic data. Fails with a clear error if
official fastMRI data is not present.
"""

import os
import sys
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml

# Add repository root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics.fft import kspace_to_image_torch, image_to_kspace_torch
from src.data.preprocessing import complex_abs, normalize_zero_one
from src.data.undersampling import RandomMaskFunc
from src.evaluation.metrics import evaluate_reconstruction
from src.visualization.reconstructions import plot_zero_filled_comparison
from src.visualization.error_maps import compute_absolute_error_map


def load_config(config_path="configs/base.yaml"):
    """Load configuration dictionary from YAML file."""
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r") as f:
            return yaml.safe_load(f)
    return {}


def find_official_h5_file(cli_path=None, config_path="configs/base.yaml"):
    """
    Search for official fastMRI HDF5 files in priority order:
    1. CLI argument (if supplied)
    2. Configured val_dir from configs/base.yaml (e.g. /kaggle/working/data/fastmri/knee_singlecoil_val)
    3. Local workspace fallback (data/fastmri/knee_singlecoil_val)
    4. Configured dataset_root (e.g. /kaggle/working/data/fastmri)
    5. Local fallback (data/fastmri)
    """
    candidates_dirs = []

    if cli_path:
        p = Path(cli_path)
        if p.is_file() and p.suffix.lower() in [".h5", ".hdf5"]:
            return p
        if p.is_dir():
            candidates_dirs.append(p)

    cfg = load_config(config_path)
    val_dir_str = cfg.get("data", {}).get("val_dir")
    if val_dir_str:
        candidates_dirs.append(Path(val_dir_str))

    # Local workspace candidates
    candidates_dirs.append(Path("data/fastmri/knee_singlecoil_val"))

    root_dir_str = cfg.get("data", {}).get("dataset_root")
    if root_dir_str:
        candidates_dirs.append(Path(root_dir_str))

    candidates_dirs.append(Path("data/fastmri"))

    # Search for first existing H5 file
    searched_paths = []
    for cdir in candidates_dirs:
        resolved = cdir.resolve()
        searched_paths.append(str(resolved))
        if cdir.exists():
            h5_files = sorted(list(cdir.glob("*.h5")) + list(cdir.glob("*.hdf5")) + list(cdir.glob("**/*.h5")))
            if h5_files:
                return h5_files[0]

    # If no files found, raise a clear error without generating synthetic data
    searched_summary = "\n  - ".join(searched_paths)
    raise FileNotFoundError(
        f"\n[ERROR] No official fastMRI HDF5 files found!\n"
        f"Searched paths:\n  - {searched_summary}\n\n"
        f"Please prepare the official NYU fastMRI dataset subset first by running:\n"
        f"  python -m src.data.download_fastmri --split val --max-files 10\n"
        f"Or set the FASTMRI_VAL_URL secret and run notebooks/01_environment.ipynb.\n"
        f"Synthetic data will NOT be generated."
    )


def run_verification(data_file_path: Path = None, config_path: str = "configs/base.yaml"):
    print("========================================")
    print("Official FastMRI Pipeline Verification")
    print("========================================")

    # Resolve official HDF5 file
    if data_file_path is None:
        data_file_path = find_official_h5_file(config_path=config_path)

    print(f"1. Selected Official H5 File: {data_file_path.name}")
    print(f"   Full Path: {data_file_path.resolve()}")
    print(f"   File Size: {data_file_path.stat().st_size / (1024 * 1024):.2f} MB")

    # 1. Inspect HDF5 Keys
    with h5py.File(data_file_path, "r") as hf:
        keys = list(hf.keys())
        print(f"2. HDF5 Internal Keys: {keys}")
        if "kspace" not in hf:
            raise KeyError(f"Expected 'kspace' key in {data_file_path.name}, but found {keys}")

        kspace_dataset = hf["kspace"]
        kspace_shape = kspace_dataset.shape
        kspace_dtype = kspace_dataset.dtype
        print(f"3. K-Space Shape: {kspace_shape} (slices, height, width)")
        print(f"   K-Space Dtype: {kspace_dtype}")

        # Pick center slice
        total_slices = kspace_shape[0]
        slice_idx = total_slices // 2
        print(f"4. Selected Slice: {slice_idx} of {total_slices}")
        raw_kspace_slice = kspace_dataset[slice_idx]

        has_esc = "reconstruction_esc" in hf
        has_rss = "reconstruction_rss" in hf
        print(f"   Ground Truth References available: ESC={has_esc}, RSS={has_rss}")

    # Convert k-space slice to complex PyTorch tensor
    kspace_tensor = torch.from_numpy(raw_kspace_slice)
    print(f"5. Converted k-space slice to PyTorch tensor: shape={kspace_tensor.shape}, dtype={kspace_tensor.dtype}")

    # Reconstruct ground truth via orthogonal centered 2D IFFT
    gt_complex = kspace_to_image_torch(kspace_tensor)
    gt_mag = complex_abs(gt_complex).numpy()
    gt_norm = normalize_zero_one(gt_mag)
    print(f"6. Reconstructed Ground-Truth Image (IFFT): shape={gt_norm.shape}, range=[{gt_norm.min():.4f}, {gt_norm.max():.4f}]")

    # Set up output directories
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    results = {}
    config = load_config(config_path)
    seed = config.get("experiment", {}).get("seed", 42)

    # Test 4x and 8x accelerations
    for accel, center_frac in [(4, 0.08), (8, 0.04)]:
        mask_func = RandomMaskFunc(center_fractions=[center_frac], accelerations=[accel], seed=seed)
        mask = mask_func(kspace_tensor.shape, seed=seed)
        y_undersampled = mask * kspace_tensor

        # Zero-Filled Inverse FFT Reconstruction
        zf_complex = kspace_to_image_torch(y_undersampled)
        zf_mag = complex_abs(zf_complex).numpy()
        zf_norm = normalize_zero_one(zf_mag)

        # Quantitative Metrics
        metrics = evaluate_reconstruction(
            gt=gt_norm,
            pred=zf_norm,
            y_measured=y_undersampled,
            x_pred_complex=zf_complex,
            mask=mask
        )

        # Error Map & Visual Plotting
        error_map = compute_absolute_error_map(gt_norm, zf_norm)
        fig_save_path = f"results/figures/verification_{accel}x_{data_file_path.stem}_slice{slice_idx}.png"

        # The plotting function now safely broadcasts 1D Cartesian masks to 2D
        plot_zero_filled_comparison(
            gt=gt_norm,
            mask=mask,
            undersampled_kspace=y_undersampled,
            zf_reconstruction=zf_norm,
            error_map=error_map,
            metrics=metrics,
            accel_factor=accel,
            save_path=fig_save_path
        )

        results[f"{accel}x"] = metrics
        print(f"\n--- Acceleration {accel}x Results ---")
        print(f"  PSNR         : {metrics['psnr']:.2f} dB")
        print(f"  SSIM         : {metrics['ssim']:.4f}")
        print(f"  NMSE         : {metrics['nmse']:.4f}")
        print(f"  K-space Error: {metrics['kspace_error']:.2e}")
        print(f"  Plot Saved   : {fig_save_path}")

    print("\n========================================")
    print("VERIFICATION COMPLETED SUCCESSFULLY")
    print(f"Validated on official fastMRI file: {data_file_path.name}")
    print("========================================")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify official fastMRI data pipeline.")
    parser.add_argument("--file", type=str, default=None, help="Specific HDF5 file path to verify")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML configuration")
    args = parser.parse_args()

    cli_file = Path(args.file) if args.file else None
    run_verification(data_file_path=cli_file, config_path=args.config)
