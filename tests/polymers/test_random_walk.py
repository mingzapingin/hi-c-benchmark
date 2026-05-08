"""Tests for random walk polymer generator."""

import numpy as np
import pytest

from hic_benchmark.polymers import random_walk


def test_returns_correct_shape() -> None:
    """Test that returned array has shape (n_beads, 3)."""
    coords = random_walk(n_beads=5)
    assert coords.shape == (5, 3)


def test_returns_float64() -> None:
    """Test that returned array has dtype float64."""
    coords = random_walk(n_beads=5)
    assert coords.dtype == np.float64


def test_first_bead_always_at_origin() -> None:
    """Test that first bead is always at (0, 0, 0)."""
    coords = random_walk(n_beads=10)
    assert np.allclose(coords[0], [0.0, 0.0, 0.0])


def test_same_seed_gives_identical_output() -> None:
    """Test that same seed produces identical results."""
    coords1 = random_walk(n_beads=10, seed=42)
    coords2 = random_walk(n_beads=10, seed=42)
    assert np.array_equal(coords1, coords2)


def test_different_seeds_give_different_output() -> None:
    """Test that different seeds produce different results."""
    coords1 = random_walk(n_beads=10, seed=42)
    coords2 = random_walk(n_beads=10, seed=123)
    assert not np.array_equal(coords1, coords2)


def test_invalid_n_beads_raises_value_error() -> None:
    """Test that invalid n_beads raises ValueError."""
    for n_beads in (0, -1, -100):
        with pytest.raises(ValueError):
            random_walk(n_beads=n_beads)


def test_step_size_matches_expected_std() -> None:
    """Test that step sizes match the expected standard deviation."""
    coords = random_walk(n_beads=10000, step_size=2.0, seed=0)
    steps = np.diff(coords, axis=0)
    assert steps.std() is not None
    assert np.isclose(steps.std(), 2.0, rtol=0.03)
