"""Compute pairwise Euclidean distances between all beads in a polymer."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.spatial.distance import pdist, squareform

from hic_benchmark.utils import Coords, SquareMatrix


def pairwise_distances(coords: Coords) -> SquareMatrix:
    """Compute pairwise Euclidean distances between all beads.

    Calculates the Euclidean distance between each pair of beads in a polymer
    structure and returns them in a square symmetric matrix format.

    Args:
        coords: A (N, 3) array of bead coordinates in 3D space.

    Returns:
        A (N, N) square matrix where element [i, j] contains the Euclidean
        distance between bead i and bead j. The matrix is symmetric with
        zeros on the diagonal.

    Raises:
        ValueError: If coords is not a 2D array with exactly 3 columns.
    """
    # Validate shape before computation
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"Expected coords to have shape (N, 3), got {coords.shape}"
        )

    # Cast input to np.float64 once near the top
    coords_float = np.asarray(coords, dtype=np.float64)

    # Use pdist and squareform to compute pairwise Euclidean distances
    distances = squareform(pdist(coords_float, metric="euclidean"))

    return cast(SquareMatrix, distances)
