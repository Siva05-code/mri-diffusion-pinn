"""
Official NYU FastMRI Dataset Downloader & Selective Extractor.

Supports streaming extraction from official NYU FastMRI signed S3 URLs (.tar.xz / .tar.gz)
directly to disk without saving the complete multi-gigabyte archive on disk.
Extracts a configurable development subset of HDF5 (.h5) files for storage-efficient
development in disk-constrained environments (such as Kaggle).

Private URLs are obtained securely via Kaggle Secrets or environment variables
(e.g., FASTMRI_VAL_URL, FASTMRI_TRAIN_URL) and are NEVER printed or logged.
"""

import os
import sys
import argparse
import urllib.request
import urllib.parse
import tarfile
import lzma
import gzip
from pathlib import Path
import yaml


def load_config(config_path="configs/base.yaml"):
    """Load configuration dictionary from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def mask_url(url: str) -> str:
    """Mask signed URL tokens, query parameters, and credentials for safe logging."""
    if not url:
        return "<empty>"
    try:
        parsed = urllib.parse.urlparse(url)
        # Display only protocol, domain, and path without query string / signature
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?***[PROTECTED SIGNATURE HIDDEN]***"
    except Exception:
        return "<protected url>"


def get_signed_url(split="val", cli_url=None):
    """
    Securely retrieve official signed fastMRI URL from environment, Kaggle Secrets, or CLI.
    Never prints or logs the raw secret value.
    """
    if cli_url:
        return cli_url.strip()

    secret_key = f"FASTMRI_{split.upper()}_URL"
    generic_key = "FASTMRI_URL"

    # 1. Check Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        val = user_secrets.get_secret(secret_key)
        if val:
            return val.strip()
        val_generic = user_secrets.get_secret(generic_key)
        if val_generic:
            return val_generic.strip()
    except Exception:
        pass  # Not running in Kaggle or secrets not set

    # 2. Check Environment Variables
    env_val = os.environ.get(secret_key) or os.environ.get(generic_key)
    if env_val:
        return env_val.strip()

    # 3. Check local .env file if present
    env_file = Path(".env")
    if env_file.exists():
        try:
            raw_urls = []
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k in [secret_key, generic_key] and v:
                            return v
                    elif line.startswith("http://") or line.startswith("https://"):
                        raw_urls.append(line)
            # If line is a direct URL matching the split
            for u in raw_urls:
                if f"knee_singlecoil_{split}" in u or "knee" in u:
                    return u
            if raw_urls and len(raw_urls) == 1:
                return raw_urls[0]
        except Exception:
            pass

    return None


def extract_from_local_archive(archive_path: Path, dest_dir: Path, max_files: int):
    """Selectively extract H5 files from a locally downloaded archive."""
    print(f"Reading local archive: {archive_path.name}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_count = 0

    mode = "r:*"
    if archive_path.name.endswith(".xz"):
        mode = "r:xz"
    elif archive_path.name.endswith(".gz"):
        mode = "r:gz"

    with tarfile.open(archive_path, mode) as tar:
        for member in tar:
            if extracted_count >= max_files:
                break
            if member.isfile() and (member.name.endswith(".h5") or member.name.endswith(".hdf5")):
                base_name = Path(member.name).name
                target_file = dest_dir / base_name
                if target_file.exists():
                    print(f"  [Skipped - exists] {base_name}")
                    extracted_count += 1
                    continue

                print(f"  [Extracting] {base_name} ({member.size / (1024*1024):.1f} MB)...")
                f_in = tar.extractfile(member)
                if f_in:
                    with open(target_file, "wb") as f_out:
                        while True:
                            chunk = f_in.read(1024 * 1024)
                            if not chunk:
                                break
                            f_out.write(chunk)
                    extracted_count += 1

    return extracted_count


def extract_from_stream(url: str, dest_dir: Path, max_files: int):
    """
    Stream signed HTTP response directly into decompression and extract only max_files H5 files.
    This avoids storing the 16+ GB or 72+ GB compressed archive on disk.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    masked = mask_url(url)
    print(f"Connecting to fastMRI download stream: {masked}")
    print(f"Target destination: {dest_dir.resolve()}")
    print(f"Subset limit: maximum {max_files} H5 files will be extracted from stream.")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FastMRI-Streaming-Client/1.0"}
    )

    extracted_count = 0
    with urllib.request.urlopen(req) as resp:
        # Determine format from URL path
        url_path = urllib.parse.urlparse(url).path.lower()
        if url_path.endswith(".xz"):
            decompressed_stream = lzma.open(resp, "rb")
        elif url_path.endswith(".gz"):
            decompressed_stream = gzip.open(resp, "rb")
        else:
            decompressed_stream = resp

        # Open tarfile in stream mode ('r|*')
        with tarfile.open(fileobj=decompressed_stream, mode="r|*") as tar:
            for member in tar:
                if extracted_count >= max_files:
                    print(f"Reached subset limit of {max_files} files. Closing network stream safely.")
                    break

                if member.isfile() and (member.name.endswith(".h5") or member.name.endswith(".hdf5")):
                    base_name = Path(member.name).name
                    target_file = dest_dir / base_name
                    if target_file.exists():
                        print(f"  [Skipped - exists] {base_name}")
                        extracted_count += 1
                        continue

                    print(f"  [Streaming & Writing] {base_name} ({member.size / (1024*1024):.1f} MB)...")
                    f_in = tar.extractfile(member)
                    if f_in:
                        with open(target_file, "wb") as f_out:
                            while True:
                                chunk = f_in.read(1024 * 1024)
                                if not chunk:
                                    break
                                f_out.write(chunk)
                        extracted_count += 1

    return extracted_count


def download_official_fastmri(split="val", max_files=None, url=None, dest_dir=None, local_archive=None):
    """
    Main function to download and extract official NYU fastMRI data subset.
    """
    config = load_config()

    if max_files is None:
        if split == "train":
            max_files = config.get("data", {}).get("num_train_files", 30)
        else:
            max_files = config.get("data", {}).get("num_val_files", 10)

    if dest_dir is None:
        if split == "train":
            dest_dir = config.get("data", {}).get("train_dir", "/kaggle/working/data/fastmri/knee_singlecoil_train")
        else:
            dest_dir = config.get("data", {}).get("val_dir", "/kaggle/working/data/fastmri/knee_singlecoil_val")

    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Official NYU FastMRI Data Setup")
    print("========================================")
    print(f"Split           : {split}")
    print(f"Target Directory: {dest_path.resolve()}")
    print(f"Max Files Subset: {max_files}")

    # Check existing files in destination directory
    existing_h5 = list(dest_path.glob("*.h5")) + list(dest_path.glob("*.hdf5"))
    if len(existing_h5) >= max_files:
        print(f"\nFound {len(existing_h5)} existing H5 files in destination directory.")
        print("Required development subset already satisfied. Skipping download.")
        print_summary(dest_path)
        return

    # Check local archive if provided
    if local_archive and os.path.exists(local_archive):
        print(f"Using local archive: {local_archive}")
        extract_from_local_archive(Path(local_archive), dest_path, max_files)
        print_summary(dest_path)
        return

    # Retrieve official signed download URL
    signed_url = get_signed_url(split=split, cli_url=url)
    if not signed_url:
        print(f"\nERROR: Official fastMRI download URL for '{split}' split is not configured!")
        print(f"\nTo access official NYU fastMRI data:")
        print(f"1. Register at: https://fastmri.med.nyu.edu/")
        print(f"2. You will receive official signed Amazon S3 download links via email.")
        print(f"3. In Kaggle, set the link as a Kaggle Secret: FASTMRI_{split.upper()}_URL")
        print(f"   Or set the environment variable: export FASTMRI_{split.upper()}_URL='<your-link>'")
        print(f"   Or pass locally via CLI: python -m src.data.download_fastmri --split {split} --url '<your-link>'")
        print("========================================")
        sys.exit(1)

    # Perform streaming extraction
    try:
        extract_from_stream(signed_url, dest_path, max_files)
    except Exception as e:
        print(f"\nError during streaming extraction: {e}")
        print("If the link has expired, request a fresh link from NYU fastMRI.")
        sys.exit(1)

    print_summary(dest_path)


def print_summary(dest_path: Path):
    """Print directory contents summary."""
    h5_files = sorted(list(dest_path.glob("*.h5")) + list(dest_path.glob("*.hdf5")))
    total_size_mb = sum(f.stat().st_size for f in h5_files) / (1024 * 1024)
    print("\n========================================")
    print("FastMRI Subset Extraction Summary")
    print("========================================")
    print(f"Directory : {dest_path.resolve()}")
    print(f"H5 Files  : {len(h5_files)}")
    print(f"Total Size: {total_size_mb:.2f} MB")
    print("Sample Files:")
    for f in h5_files[:5]:
        print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official NYU fastMRI selective dataset extractor.")
    parser.add_argument("--split", choices=["val", "train"], default="val", help="Data split to download (val or train)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of H5 files to extract (default from config)")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory path")
    parser.add_argument("--url", type=str, default=None, help="Direct signed URL (alternative to Kaggle Secret / env var)")
    parser.add_argument("--local-archive", type=str, default=None, help="Path to a local .tar.xz / .tar.gz archive file")

    args = parser.parse_args()
    download_official_fastmri(
        split=args.split,
        max_files=args.max_files,
        url=args.url,
        dest_dir=args.dest,
        local_archive=args.local_archive
    )

