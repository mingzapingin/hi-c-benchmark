"""Smoke tests. Confirm the package imports and basic utilities work."""

from __future__ import annotations

import numpy as np

import hic_benchmark
from hic_benchmark.utils import set_seed


def test_package_has_version() -> None:
    assert isinstance(hic_benchmark.__version__, str)
    assert hic_benchmark.__version__ != ""


def test_set_seed_returns_generator() -> None:
    rng = set_seed(0)
    assert isinstance(rng, np.random.Generator)


def test_set_seed_is_reproducible() -> None:
    rng1 = set_seed(42)
    rng2 = set_seed(42)
    assert np.array_equal(rng1.standard_normal(10), rng2.standard_normal(10))
