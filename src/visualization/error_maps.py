"""
Error Map Visualization Utilities for MRI Reconstruction.
"""

import numpy as np


def compute_absolute_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute point-wise absolute error map |gt - pred|."""
    return np.abs(gt - pred)


def compute_squared_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute point-wise squared error map (gt - pred)^2."""
    return (gt - pred) ** 2

