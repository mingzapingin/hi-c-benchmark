"""Bead-spring polymer generator."""

from __future__ import annotations

import numpy as np

from ..utils import Coords


def bead_spring(
    n_beads: int,
    bond_length: float = 1.0,
    seed: int | None = None,
) -> Coords:
    """Generate a bead-spring polymer chain.

    Creates a polymer chain where consecutive beads are connected with fixed
    bond lengths. The first bead is fixed at the origin, and subsequent
    beads are placed by generating random directions and scaling them to
    the specified bond length.

    Args:
        n_beads: Number of beads in the chain (must be >= 1).
        bond_length: Target distance between consecutive beads (must be > 0).
        seed: Random seed for reproducibility.

    Returns:
        Coords array of shape (n_beads, 3) with dtype float64.
        First bead is at origin (0, 0, 0).

    Raises:
        ValueError: If n_beads < 1 or bond_length <= 0.
    """
    if n_beads < 1:
        raise ValueError(f"n_beads must be >= 1, got {n_beads}")

    if bond_length <= 0:
        raise ValueError(f"bond_length must be > 0, got {bond_length}")

    rng = np.random.default_rng(seed)

    # First bead at origin
    coords = np.zeros((n_beads, 3), dtype=np.float64)

    # Draw raw directions for remaining beads
    directions = rng.standard_normal((n_beads - 1, 3))

    # Normalize each row to unit length
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_directions = directions / norms

    # Scale to bond_length
    displacements = unit_directions * bond_length

    # Cumulative sum gives positions relative to first bead
    relative_positions = np.cumsum(displacements, axis=0)

    # Vstack origin at the top
    coords[1:] = relative_positions

    return coords
