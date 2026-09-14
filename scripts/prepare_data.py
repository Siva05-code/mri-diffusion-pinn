"""
FastMRI Data Preparation CLI Script.

Wraps src.data.download_fastmri to selectively extract official NYU fastMRI H5 subsets.
"""

import sys
import os
import argparse

# Add repository root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.download_fastmri import download_official_fastmri


def main():
    parser = argparse.ArgumentParser(description="Prepare official NYU fastMRI dataset development subset.")
    parser.add_argument("--split", choices=["val", "train"], default="val", help="Data split to download (val or train)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of H5 files to extract")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory path")
    parser.add_argument("--url", type=str, default=None, help="Direct signed URL (alternative to Kaggle Secret / env var)")
    parser.add_argument("--local-archive", type=str, default=None, help="Path to local .tar.xz / .tar.gz archive file")

    args = parser.parse_args()
    download_official_fastmri(
        split=args.split,
        max_files=args.max_files,
        url=args.url,
        dest_dir=args.dest,
        local_archive=args.local_archive
    )


if __name__ == "__main__":
    main()

