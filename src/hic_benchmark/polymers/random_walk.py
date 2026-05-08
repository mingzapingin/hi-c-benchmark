"""Random walk polymer generator."""


import numpy as np

from ..utils import Coords


def random_walk(
    n_beads: int,
    step_size: float = 1.0,
    seed: int | None = None,
) -> Coords:
    """Generate a random walk polymer chain.

    Args:
        n_beads: Number of beads in the chain (must be >= 1).
        step_size: Standard deviation for each step (must be > 0).
        seed: Random seed for reproducibility.

    Returns:
        Coords array of shape (n_beads, 3) with dtype float64.
        First bead is at origin (0, 0, 0).

    Raises:
        ValueError: If n_beads < 1 or step_size <= 0.
    """
    if n_beads < 1:
        raise ValueError(f"n_beads must be >= 1, got {n_beads}")

    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")

    rng = np.random.default_rng(seed)

    # First bead at origin
    coords = np.zeros((n_beads, 3), dtype=np.float64)

    # Draw steps for remaining beads
    steps = rng.normal(size=(n_beads - 1, 3), scale=step_size)

    # Cumulative sum gives final positions
    coords[1:] = np.cumsum(steps, axis=0)

    return coords
