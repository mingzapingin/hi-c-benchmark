"""Shared utilities. Seeding, type aliases, small helpers."""

from __future__ import annotations

import os
import random
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# --- Type aliases used across the package ---------------------------------

#: Bead coordinate array, shape (N, 3).
Coords: TypeAlias = NDArray[np.float64]

#: Square symmetric distance or contact matrix, shape (N, N).
SquareMatrix: TypeAlias = NDArray[np.float64]


# --- Seeding --------------------------------------------------------------


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and the hash environment for reproducibility.

    Returns a `numpy.random.Generator` that downstream callers should thread
    through their stochastic functions instead of touching global state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # for legacy code paths only
    return np.random.default_rng(seed)
