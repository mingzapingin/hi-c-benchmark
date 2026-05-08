"""Tests for bead-spring polymer generator."""

import numpy as np
import pytest

from hic_benchmark.polymers import bead_spring


def test_returns_correct_shape() -> None:
    """Test that returned array has shape (n_beads, 3)."""
    coords = bead_spring(n_beads=5)
    assert coords.shape == (5, 3)


def test_returns_float64() -> None:
    """Test that returned array has dtype float64."""
    coords = bead_spring(n_beads=5)
    assert coords.dtype == np.float64


def test_single_bead_is_at_origin() -> None:
    """Test that single bead is at origin."""
    coords = bead_spring(n_beads=1)
    assert np.allclose(coords, [0.0, 0.0, 0.0])


def test_first_bead_always_at_origin() -> None:
    """Test that first bead is always at (0, 0, 0)."""
    coords = bead_spring(n_beads=10)
    assert np.allclose(coords[0], [0.0, 0.0, 0.0])


def test_all_bonds_have_exact_target_length() -> None:
    """Test that all consecutive bead pairs have exactly bond_length distance."""
    coords = bead_spring(n_beads=10, bond_length=2.0, seed=42)
    for i in range(1, len(coords)):
        bond_length = np.linalg.norm(coords[i] - coords[i - 1])
        assert np.isclose(bond_length, 2.0)


def test_bond_length_parameter_is_respected() -> None:
    """Test that bond_length parameter is correctly applied."""
    for target in (0.5, 1.0, 3.0, 7.5):
        coords = bead_spring(n_beads=10, bond_length=target, seed=42)
        for i in range(1, len(coords)):
            bond_length = np.linalg.norm(coords[i] - coords[i - 1])
            assert np.isclose(bond_length, target)


def test_same_seed_gives_identical_output() -> None:
    """Test that same seed produces identical results."""
    coords1 = bead_spring(n_beads=10, seed=42)
    coords2 = bead_spring(n_beads=10, seed=42)
    assert np.array_equal(coords1, coords2)


def test_different_seeds_give_different_output() -> None:
    """Test that different seeds produce different results."""
    coords1 = bead_spring(n_beads=10, seed=42)
    coords2 = bead_spring(n_beads=10, seed=123)
    assert not np.array_equal(coords1, coords2)


@pytest.mark.parametrize("n_beads", [0, -1, -100])
def test_invalid_n_beads_raises(n_beads: int) -> None:
    """Test that invalid n_beads raises ValueError."""
    with pytest.raises(ValueError):
        bead_spring(n_beads=n_beads)


@pytest.mark.parametrize("bond_length", [0.0, -1.0])
def test_invalid_bond_length_raises(bond_length: float) -> None:
    """Test that invalid bond_length raises ValueError."""
    with pytest.raises(ValueError):
        bead_spring(n_beads=10, bond_length=bond_length)


def test_step_directions_are_isotropic() -> None:
    """Test that step directions are isotropic (no preferred direction)."""
    coords = bead_spring(n_beads=10_000, bond_length=1.0, seed=0)
    steps = np.diff(coords, axis=0)
    assert np.all(np.abs(steps.mean(axis=0)) < 0.05)
