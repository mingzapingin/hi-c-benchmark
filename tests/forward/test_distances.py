"""Tests for pairwise_distances."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from hic_benchmark.forward import pairwise_distances
from hic_benchmark.polymers.bead_spring import bead_spring
from hic_benchmark.polymers.random_walk import random_walk


def test_returns_square_matrix():
    """Verify output is a square (N, N) matrix."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    result = pairwise_distances(coords)
    assert result.shape == (2, 2)


def test_returns_float64_even_for_int_input():
    """Verify output dtype is float64 even when input has int dtype."""
    coords = np.array([[0, 0, 0], [3, 0, 0]], dtype=int)
    result = pairwise_distances(coords)
    assert result.dtype == np.float64


def test_single_bead_returns_one_by_one_zero():
    """Single bead returns a 1x1 matrix with zero distance."""
    coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    result = pairwise_distances(coords)
    assert result.shape == (1, 1)
    assert_array_almost_equal(result, [[0.0]])


def test_known_distance_for_two_beads():
    """Verify known distance between two points."""
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float64)
    result = pairwise_distances(coords)
    # Expected distance = sqrt(3^2 + 4^2) = 5.0
    assert_array_almost_equal(result[0, 1], 5.0, decimal=10)


def test_diagonal_is_zero():
    """Verify diagonal elements are zero (distance from a point to itself)."""
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    result = pairwise_distances(coords)
    assert_array_almost_equal(np.diag(result), [0.0, 0.0, 0.0])


def test_matrix_is_symmetric():
    """Verify distance matrix is symmetric."""
    coords = np.random.default_rng(0).random((10, 3))
    result = pairwise_distances(coords)
    assert_array_almost_equal(result, result.T)


def test_all_distances_nonnegative():
    """Verify all distances are nonnegative."""
    coords = np.random.default_rng(42).random((20, 3))
    result = pairwise_distances(coords)
    assert np.all(result >= 0.0)


def test_translation_invariance():
    """Verify distances are invariant to translation of all points."""
    rng = np.random.default_rng(0)
    n_beads = 20
    coords = rng.random((n_beads, 3))
    offset = rng.random((3,))
    coords_translated = coords + offset
    result_original = pairwise_distances(coords)
    result_translated = pairwise_distances(coords_translated)
    assert_array_almost_equal(result_original, result_translated)


def test_random_walk_distances_are_finite_and_well_formed():
    """Verify random_walk produces finite, well-formed distances."""
    coords = random_walk(n_beads=30, seed=0)
    D = pairwise_distances(coords)
    assert D.shape == (30, 30)
    assert D.dtype == np.float64
    assert np.all(np.isfinite(D))
    assert np.all(np.diag(D) == 0.0)
    assert np.all(D == D.T)


def test_bead_spring_super_diagonal_equals_bond_length():
    """Verify super-diagonal equals the specified bond_length."""
    coords = bead_spring(n_beads=100, bond_length=2.5, seed=0)
    D = pairwise_distances(coords)
    super_diag = np.diag(D, k=1)
    assert np.allclose(super_diag, 2.5, rtol=1e-10)


@pytest.mark.parametrize(
    "bad_shape",
    [(10,), (10, 2), (10, 4), (10, 3, 1)],
    ids=["1d", "2d-too-narrow", "2d-too-wide", "3d"],
)
def test_invalid_shape_raises(bad_shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        pairwise_distances(np.zeros(bad_shape))
