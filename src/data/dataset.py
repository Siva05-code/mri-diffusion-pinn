"""
FastMRI HDF5 PyTorch Dataset Wrapper.

Loads raw k-space HDF5 files, parses slices, and formats complex PyTorch tensors.
"""

import os
from pathlib import Path
from typing import Optional, Callable
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FastMRIDataset(Dataset):
    """
    PyTorch Dataset for single-coil fastMRI raw k-space HDF5 files.
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.examples = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data directory '{self.root_dir}' does not exist.")

        # Index all .h5 / .hdf5 files and their slice counts
        h5_files = sorted(list(self.root_dir.glob("*.h5")) + list(self.root_dir.glob("*.hdf5")))
        for fname in h5_files:
            try:
                with h5py.File(fname, "r") as hf:
                    if "kspace" in hf:
                        num_slices = hf["kspace"].shape[0]
                        for slice_idx in range(num_slices):
                            self.examples.append((fname, slice_idx))
            except Exception as e:
                print(f"Warning: Skipping file {fname} due to read error: {e}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        fname, slice_idx = self.examples[idx]

        with h5py.File(fname, "r") as hf:
            kspace_raw = hf["kspace"][slice_idx]
            target = hf["reconstruction_rss"][slice_idx] if "reconstruction_rss" in hf else (
                hf["reconstruction_esc"][slice_idx] if "reconstruction_esc" in hf else None
            )
            attrs = dict(hf.attrs)

        # Convert complex numpy kspace to complex torch tensor or (height, width, 2)
        if np.iscomplexobj(kspace_raw):
            kspace_tensor = torch.from_numpy(kspace_raw)
        else:
            kspace_tensor = torch.from_numpy(kspace_raw.astype(np.float32))

        sample = {
            "kspace": kspace_tensor,
            "filename": fname.name,
            "slice_idx": slice_idx,
            "attrs": attrs
        }

        if target is not None:
            sample["target"] = torch.from_numpy(target.astype(np.float32))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

