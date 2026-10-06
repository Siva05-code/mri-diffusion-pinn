"""
Dataset for physics-guided conditional diffusion MRI reconstruction.

Two backends are supported and both produce identical sample dictionaries:

1. ``raw``   -- reads fastMRI single-coil HDF5 volumes directly.
2. ``cache`` -- reads a pre-built ``.npy`` memmap of 320x320 targets produced by
                ``scripts/prepare_data.py``. Strongly preferred on Colab: the
                per-slice IFFT of a 640x372 array is the single slowest part of
                the raw path and it is pure waste to repeat it every epoch.

Each item is:

    target           (1, H, W)  ground-truth magnitude image in [0, 1]
    zf               (1, H, W)  zero-filled reconstruction, real
    residual         (1, H, W)  (target - zf) / residual_scale, clipped to [-1, 1]
    mask             (1, H, W)  sampling mask, values in {0, 1}
    measured_kspace  (1, H, W)  complex64, mask * FFT(target)
    scale            ()         intensity scale used for normalisation
    acceleration     ()         acceleration factor actually used
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import center_crop, normalize_magnitude, kspace_to_target
from src.data.undersampling import make_cartesian_mask


def _fft2c_np(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))


def _ifft2c_np(k: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k), norm="ortho"))


class FastMRIDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 320,
        accelerations=(4,),
        center_fraction: float = 0.08,
        mask_mode: str = "random",
        residual_scale: float = 1.0,
        max_files: int = 0,
        max_slices: int = 0,
        slice_range: tuple[float, float] = (0.15, 0.95),
        training: bool = True,
        seed: int = 42,
        cache: str | None = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.accelerations = list(accelerations) if isinstance(accelerations, (list, tuple)) else [accelerations]
        self.center_fraction = center_fraction
        self.mask_mode = mask_mode
        self.residual_scale = float(residual_scale)
        self.training = training
        self.seed = seed

        self.cache_targets = None
        cache_path = Path(cache) if cache else None

        if cache_path is not None and cache_path.exists():
            # ---- cached backend -------------------------------------------
            self.mode = "cache"
            self.cache_targets = np.load(cache_path, mmap_mode="r")
            if self.cache_targets.shape[-1] != image_size:
                raise ValueError(
                    f"Cache was built at size {self.cache_targets.shape[-1]} but "
                    f"image_size={image_size} was requested. Rebuild the cache."
                )
            n = self.cache_targets.shape[0]
            self.index = list(range(n))
            if max_slices > 0:
                self.index = self.index[:max_slices]
        else:
            # ---- raw HDF5 backend -----------------------------------------
            self.mode = "raw"
            self.root = Path(root)
            files = sorted(
                list(self.root.rglob("*.h5")) + list(self.root.rglob("*.hdf5"))
            )
            if max_files > 0:
                files = files[:max_files]
            if not files:
                raise FileNotFoundError(
                    f"No .h5 files found under {self.root}.\n"
                    "Point --train_dir / --val_dir at a folder containing fastMRI "
                    "single-coil knee HDF5 volumes, or build a cache first with "
                    "scripts/prepare_data.py."
                )

            self.files = files
            self.index = []
            for path in files:
                with h5py.File(path, "r") as f:
                    key = "reconstruction_esc" if "reconstruction_esc" in f else "kspace"
                    if key not in f:
                        continue
                    n_slices = int(f[key].shape[0])
                # Outer slices are mostly noise and drag the metrics around.
                lo = int(n_slices * slice_range[0])
                hi = int(n_slices * slice_range[1])
                lo, hi = max(lo, 0), max(min(hi, n_slices), 1)
                for s in range(lo, hi):
                    self.index.append((path, s))

            if max_slices > 0:
                self.index = self.index[:max_slices]

        if not self.index:
            raise RuntimeError("Dataset is empty after filtering.")

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------
    def _load_target(self, i: int) -> tuple[np.ndarray, float]:
        if self.mode == "cache":
            return np.asarray(self.cache_targets[self.index[i]], dtype=np.float32), 1.0

        path, slice_idx = self.index[i]
        with h5py.File(path, "r") as f:
            if "reconstruction_esc" in f:
                # fastMRI ships the official 320x320 emulated single-coil
                # reconstruction. Using it removes any ambiguity about crop
                # conventions between this code and the benchmark.
                img = np.asarray(f["reconstruction_esc"][slice_idx], dtype=np.float32)
                img = center_crop(img, self.image_size)
                return normalize_magnitude(img)
            k = np.asarray(f["kspace"][slice_idx])
        return kspace_to_target(k, self.image_size)

    def _rng(self, i: int) -> np.random.Generator:
        if self.training:
            # Fresh mask every time this slice is drawn: the model must not be
            # able to memorise one mask per slice.
            return np.random.default_rng()
        # Validation and test masks are fixed so that every checkpoint and every
        # ablation row is scored on exactly the same undersampling pattern.
        return np.random.default_rng(self.seed + i)

    # ------------------------------------------------------------------
    def __getitem__(self, i: int):
        target_np, scale = self._load_target(i)
        rng = self._rng(i)

        accel = int(self.accelerations[rng.integers(len(self.accelerations))])
        mask_np = make_cartesian_mask(
            self.image_size,
            accel,
            self.center_fraction,
            rng,
            mode=self.mask_mode,
        )

        full_k = _fft2c_np(target_np)
        measured_k = (mask_np * full_k).astype(np.complex64)
        zf_np = _ifft2c_np(measured_k).real.astype(np.float32)

        target = torch.from_numpy(target_np).unsqueeze(0)
        zf = torch.from_numpy(zf_np).unsqueeze(0)
        mask = torch.from_numpy(mask_np).unsqueeze(0)
        measured = torch.from_numpy(measured_k).unsqueeze(0)

        residual = (target - zf) / max(self.residual_scale, 1e-6)
        residual = residual.clamp(-1.0, 1.0)

        return {
            "target": target,
            "zf": zf,
            "residual": residual,
            "mask": mask,
            "measured_kspace": measured,
            "scale": torch.tensor(scale, dtype=torch.float32),
            "acceleration": torch.tensor(accel, dtype=torch.long),
        }


def estimate_residual_scale(
    dataset: "FastMRIDataset",
    num_samples: int = 200,
    percentile: float = 99.5,
) -> float:
    """
    Estimate the typical magnitude of (target - zero_filled).

    This is the most important piece of calibration in the whole project.
    A diffusion model is trained on x_0 samples whose scale is order 1. The
    raw residual here has a standard deviation of roughly 0.02-0.06 on a
    [0, 1] image, so feeding it in unscaled buries the signal far below the
    injected noise at almost every timestep: the network learns the trivial
    solution eps_pred = x_t, the diffusion loss collapses to a tiny number
    that looks great in the log, and the sampler returns the zero-filled
    image back. Dividing by this scale puts the residual on the same footing
    as a standard diffusion target.
    """
    saved = dataset.residual_scale
    dataset.residual_scale = 1.0
    try:
        n = min(num_samples, len(dataset))
        step = max(1, len(dataset) // n)
        values = []
        for i in range(0, len(dataset), step):
            if len(values) >= n:
                break
            item = dataset[i]
            r = (item["target"] - item["zf"]).abs().numpy()
            values.append(np.percentile(r, percentile))
    finally:
        dataset.residual_scale = saved

    scale = float(np.mean(values))
    return max(scale, 1e-3)
