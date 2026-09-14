"""
Kaggle Dataset Downloader for FastMRI Reconstruction Project.

Downloads the configured fastMRI dataset from Kaggle via the official Kaggle CLI.
Validates Kaggle authentication, handles dataset extraction, avoids duplicate downloads,
and prints a dataset summary upon completion.
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import yaml


def load_config(config_path="configs/base.yaml"):
    """Load configuration dictionary from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def check_kaggle_cli():
    """Verify that Kaggle CLI is installed and available in system PATH."""
    try:
        res = subprocess.run(["kaggle", "--version"], capture_output=True, text=True, check=True)
        return True, res.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return False, "Kaggle CLI is not available in PATH."


def check_kaggle_auth():
    """Verify Kaggle CLI authentication credentials securely without printing secrets."""
    # Kaggle CLI checks environment variables KAGGLE_USERNAME and KAGGLE_KEY, or ~/.kaggle/kaggle.json
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    has_file = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    
    if not (has_env or has_file):
        return False, "Kaggle credentials not detected in environment or ~/.kaggle/kaggle.json."
    
    try:
        res = subprocess.run(
            ["kaggle", "datasets", "list", "--mine", "--max-size", "1"],
            capture_output=True,
            text=True
        )
        if res.returncode == 0:
            return True, "Kaggle API authenticated successfully."
        else:
            return False, "Kaggle CLI authentication failed (invalid credentials)."
    except Exception as e:
        return False, f"Error validating Kaggle authentication: {e}"


def get_dataset_files_info(data_dir):
    """Scan dataset directory and return file statistics."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return [], [], 0

    all_files = [f for f in data_path.rglob("*") if f.is_file()]
    h5_files = [f for f in all_files if f.suffix.lower() in [".h5", ".hdf5"]]
    total_size_bytes = sum(f.stat().st_size for f in all_files)
    return all_files, h5_files, total_size_bytes


def download_dataset(dataset_slug=None, destination_dir=None, config_path="configs/base.yaml", force=False):
    """Download and extract the configured Kaggle dataset."""
    config = load_config(config_path)

    # 1. Resolve dataset slug
    if not dataset_slug:
        dataset_slug = os.environ.get("KAGGLE_DATASET_SLUG") or config.get("data", {}).get("kaggle_dataset", "")

    dataset_slug = dataset_slug.strip()
    if not dataset_slug:
        print("ERROR: Dataset slug is empty. Please set 'kaggle_dataset' in configs/base.yaml or set KAGGLE_DATASET_SLUG environment variable.")
        sys.exit(1)

    # 2. Resolve target destination directory
    if not destination_dir:
        destination_dir = config.get("data", {}).get("data_dir", "/kaggle/working/data/fastmri")

    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Kaggle Dataset Setup")
    print("========================================")
    print(f"Dataset:\n    {dataset_slug}")
    print(f"Destination:\n    {dest_path.resolve()}")

    # 3. Check CLI and Authentication
    cli_ok, cli_msg = check_kaggle_cli()
    if not cli_ok:
        print(f"ERROR: {cli_msg}")
        sys.exit(1)

    auth_ok, auth_msg = check_kaggle_auth()
    if not auth_ok:
        print(f"ERROR: {auth_msg}")
        sys.exit(1)

    # 4. Check if dataset already downloaded
    all_files, h5_files, total_size = get_dataset_files_info(dest_path)
    if len(h5_files) > 0 and not force:
        print("\nDataset already present. Skipping download (use force=True to re-download).")
    else:
        print("\nInitiating Kaggle CLI dataset download...")
        cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest_path), "--unzip"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Download and extraction complete.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR downloading dataset: {e.stderr.strip()}")
            sys.exit(1)

        # Refresh file statistics
        all_files, h5_files, total_size = get_dataset_files_info(dest_path)

    # 5. Output Summary
    total_size_mb = total_size / (1024 * 1024)
    example_names = [f.name for f in h5_files[:5]]

    print("\nFiles found:")
    print(f"    Total files: {len(all_files)}")
    print("\nHDF5 files:")
    print(f"    Count: {len(h5_files)}")
    print("\nTotal size:")
    print(f"    {total_size_mb:.2f} MB ({total_size_mb / 1024:.2f} GB)")
    print("\nExample files:")
    for fname in example_names:
        print(f"    {fname}")
    print("========================================")
    print("Dataset setup complete.")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download fastMRI dataset from Kaggle.")
    parser.add_argument("--slug", type=str, default=None, help="Kaggle dataset slug (OWNER/DATASET-NAME)")
    parser.add_argument("--dest", type=str, default=None, help="Target download directory")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML configuration file")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")

    args = parser.parse_args()
    download_dataset(dataset_slug=args.slug, destination_dir=args.dest, config_path=args.config, force=args.force)

