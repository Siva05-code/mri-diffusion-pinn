"""
K-Space Undersampling Mask Generators.

Implements random and equispaced 1D Cartesian undersampling masks for accelerated MRI (e.g. 4x, 8x).
Generates acquisition masks in the raw k-space domain [..., num_cols] that broadcast
across the readout dimension [..., H_raw, W_raw], as well as reconstruction-domain
conditioning masks [..., H_recon, W_recon].
"""

from typing import Tuple, Union, Optional
import torch
import numpy as np


class RandomMaskFunc:
    """
    Random 1D Cartesian undersampling mask generator.
    Preserves central low-frequency k-space region (center_fraction) and randomly samples remaining lines.
    """
    def __init__(self, center_fractions: list = [0.08], accelerations: list = [4], seed: Optional[int] = None):
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState(seed)

    def __call__(self, shape: tuple, seed: Optional[int] = None) -> torch.Tensor:
        if len(shape) < 2:
            raise ValueError(f"Shape must have at least 2 dimensions, got {shape}")
        
        num_cols = shape[-1]
        if seed is not None:
            state = self.rng.get_state()
            self.rng.seed(seed)

        num_center_lines = int(round(num_cols * self.center_fractions[0]))
        acceleration = self.accelerations[0]

        # Calculate number of low-frequency lines and high-frequency target lines
        num_low_freq = num_center_lines
        num_high_freq = int(round(num_cols / acceleration)) - num_low_freq

        # Generate center indices
        center_start = (num_cols - num_center_lines) // 2
        center_end = center_start + num_center_lines

        # Generate mask
        mask = np.zeros(num_cols, dtype=np.float32)
        mask[center_start:center_end] = 1.0

        # Sample outer k-space
        outer_indices = list(range(0, center_start)) + list(range(center_end, num_cols))
        if num_high_freq > 0 and len(outer_indices) > 0:
            selected_outer = self.rng.choice(outer_indices, size=min(num_high_freq, len(outer_indices)), replace=False)
            mask[selected_outer] = 1.0

        if seed is not None:
            self.rng.set_state(state)

        # Reshape mask to broadcast across 2D/3D shapes [1, 1, ..., num_cols]
        mask_shape = [1] * (len(shape) - 1) + [num_cols]
        return torch.from_numpy(mask.reshape(mask_shape))


class EquispacedMaskFunc:
    """
    Equispaced 1D Cartesian undersampling mask generator (fastMRI standard).
    Preserves central low-frequency k-space region (center_fraction) and uniformly samples remaining outer lines.
    """
    def __init__(self, center_fractions: list = [0.08], accelerations: list = [4], seed: Optional[int] = None):
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.seed = seed

    def __call__(self, shape: tuple, seed: Optional[int] = None) -> torch.Tensor:
        if len(shape) < 2:
            raise ValueError(f"Shape must have at least 2 dimensions, got {shape}")

        num_cols = shape[-1]
        num_center_lines = int(round(num_cols * self.center_fractions[0]))
        acceleration = self.accelerations[0]

        center_start = (num_cols - num_center_lines) // 2
        center_end = center_start + num_center_lines

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[center_start:center_end] = 1.0

        # Deterministic equispaced offset based on seed if provided
        active_seed = seed if seed is not None else self.seed
        offset = 0 if active_seed is None else (active_seed % acceleration)

        for col in range(offset, num_cols, acceleration):
            mask[col] = 1.0

        mask_shape = [1] * (len(shape) - 1) + [num_cols]
        return torch.from_numpy(mask.reshape(mask_shape))


def create_mask_func(
    mask_type: str = "equispaced",
    center_fraction: float = 0.08,
    acceleration: int = 4,
    seed: Optional[int] = None,
) -> Union[EquispacedMaskFunc, RandomMaskFunc]:
    """Factory function for creating Cartesian undersampling mask generators."""
    mask_type = mask_type.lower()
    if mask_type in ["equispaced", "regular"]:
        return EquispacedMaskFunc(
            center_fractions=[center_fraction],
            accelerations=[acceleration],
            seed=seed,
        )
    elif mask_type in ["random", "gaussian"]:
        return RandomMaskFunc(
            center_fractions=[center_fraction],
            accelerations=[acceleration],
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown mask_type: '{mask_type}'. Choose 'equispaced' or 'random'.")


def get_recon_conditioning_mask(
    mask_raw: torch.Tensor,
    recon_size: Tuple[int, int] = (320, 320),
) -> torch.Tensor:
    """
    Constructs a 2D conditioning mask aligned with the reconstruction domain [B, 1, H_recon, W_recon].
    
    In fastMRI, the 1D phase-encoding mask is applied to columns of the raw acquisition.
    To provide the neural network with explicit spatial knowledge of sampling density,
    this function constructs a [1, 1, H_recon, W_recon] tensor representing the phase-encoding
    sampling pattern resampled or center-cropped to the reconstruction FOV.

    Args:
        mask_raw: Raw acquisition mask with shape [..., W_raw].
        recon_size: Spatial dimensions of reconstruction FOV (H_recon, W_recon).

    Returns:
        torch.Tensor: 4D conditioning mask [1, 1, H_recon, W_recon] with values in {0, 1}.
    """
    while mask_raw.ndim > 1 and mask_raw.shape[-2] == 1:
        mask_raw = mask_raw.squeeze(-2)
    if mask_raw.ndim > 1:
        # Collapse extra leading dimensions to 1D phase-encoding vector
        mask_1d = mask_raw.flatten()[-mask_raw.shape[-1]:]
    else:
        mask_1d = mask_raw

    H_recon, W_recon = recon_size
    W_raw = mask_1d.shape[-1]

    if W_raw == W_recon:
        recon_1d = mask_1d
    elif W_raw > W_recon:
        # Center-crop phase-encoding lines to reconstruction window
        start = (W_raw - W_recon) // 2
        recon_1d = mask_1d[start : start + W_recon]
    else:
        # Pad symmetrically if raw width is smaller
        pad_left = (W_recon - W_raw) // 2
        pad_right = W_recon - W_raw - pad_left
        recon_1d = torch.nn.functional.pad(mask_1d, (pad_left, pad_right), value=0.0)

    # Broadcast 1D column mask across H_recon rows: [1, 1, H_recon, W_recon]
    recon_2d = recon_1d.unsqueeze(0).expand(H_recon, -1)
    return recon_2d.unsqueeze(0).unsqueeze(0).float()
