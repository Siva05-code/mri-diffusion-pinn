"""
K-Space Undersampling Mask Generators.

Implements random and equispaced 1D cartesian undersampling masks for accelerated MRI (e.g. 4x, 8x).
"""

import torch
import numpy as np


class RandomMaskFunc:
    """
    Random 1D Cartesian undersampling mask generator.
    Preserves central low-frequency k-space region (center_fraction) and randomly samples remaining lines.
    """
    def __init__(self, center_fractions: list = [0.08], accelerations: list = [4], seed: int = None):
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState(seed)

    def __call__(self, shape: tuple, seed: int = None) -> torch.Tensor:
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

