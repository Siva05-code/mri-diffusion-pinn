"""
Cartesian undersampling masks for the 320x320 reconstruction domain.

Why the masks are Hermitian-symmetric
-------------------------------------
The target image in this project is a real-valued magnitude image, so its
spectrum satisfies K[i] = conj(K[(N - i) % N]) in fftshift-ed index space.

If the sampling mask does not respect that symmetry, the undersampled
spectrum is no longer the spectrum of any real image, the zero-filled
reconstruction has a non-trivial imaginary part, and taking ``.real``
silently destroys part of the measured information. Data consistency then
stops being an exact projection and the reported DC error can never reach 0.

Mirroring every sampled phase-encode line about the centre fixes this: the
zero-filled image is exactly real and data consistency becomes a true
projection operator. The effective acceleration is preserved because the
target line count is enforced *after* mirroring.
"""

from __future__ import annotations

import numpy as np


def _symmetrise(mask_1d: np.ndarray) -> np.ndarray:
    """Enforce m[i] == m[(N - i) % N] (index 0 and N/2 are self-paired)."""
    n = mask_1d.shape[0]
    idx = np.arange(n)
    partner = (n - idx) % n
    return np.maximum(mask_1d, mask_1d[partner])


def make_cartesian_mask(
    size: int,
    acceleration: int,
    center_fraction: float,
    rng: np.random.Generator,
    mode: str = "random",
) -> np.ndarray:
    """
    Build a 1-D phase-encode mask and broadcast it over the readout direction.

    Parameters
    ----------
    size:            image / k-space size (square).
    acceleration:    nominal acceleration factor (4 or 8).
    center_fraction: fraction of lines kept in the fully-sampled centre.
    mode:            "random" (variable-density style) or "equispaced".

    Returns
    -------
    float32 array of shape (size, size), values in {0, 1}.
    """
    if acceleration < 1:
        raise ValueError("acceleration must be >= 1")

    num_low = max(2, int(round(size * center_fraction)))
    target_lines = max(num_low, int(round(size / acceleration)))

    mask_1d = np.zeros(size, dtype=np.float32)

    # Fully sampled centre, symmetric about the DC bin at index size // 2.
    center = size // 2
    half = num_low // 2
    lo = max(0, center - half)
    hi = min(size, center + half + 1)
    mask_1d[lo:hi] = 1.0

    mask_1d = _symmetrise(mask_1d)

    # Add outer lines in conjugate pairs until the budget is reached.
    candidates = np.flatnonzero(mask_1d == 0)
    if candidates.size:
        if mode == "equispaced":
            n_pairs = max(0, (target_lines - int(mask_1d.sum())) // 2)
            if n_pairs > 0:
                upper = candidates[candidates > center]
                if upper.size:
                    step = max(1, upper.size // max(n_pairs, 1))
                    picks = upper[::step][:n_pairs]
                    mask_1d[picks] = 1.0
        else:
            order = rng.permutation(candidates)
            for idx in order:
                if int(mask_1d.sum()) >= target_lines:
                    break
                mask_1d[idx] = 1.0
                mask_1d[(size - idx) % size] = 1.0

    mask_1d = _symmetrise(mask_1d)
    return np.repeat(mask_1d[None, :], size, axis=0).astype(np.float32)


def effective_acceleration(mask: np.ndarray) -> float:
    """Actual acceleration achieved by a mask (size / number of sampled lines)."""
    line = mask[0] if mask.ndim == 2 else mask
    sampled = float(line.sum())
    return float(line.shape[0]) / max(sampled, 1.0)
