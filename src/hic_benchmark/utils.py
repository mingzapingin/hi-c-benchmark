"""Shared utilities. Seeding, type aliases, small helpers."""

from __future__ import annotations

import os
import random

import numpy as np
from numpy.typing import NDArray

# --- Type aliases used across the package ---------------------------------

#: Bead coordinate array, shape (N, 3).
type Coords = NDArray[np.float64]

#: Square symmetric distance or contact matrix, shape (N, N).
type SquareMatrix = NDArray[np.float64]


# --- Seeding --------------------------------------------------------------


def set_seed(seed: int) -> np.random.Generator:
    """Set Python-level deterministic state and return a NumPy random generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    return np.random.default_rng(seed)
