# CLAUDE.md

Persistent context for Claude (and other AI coding assistants) when working in
this repository. Read this first.

## Project summary

`hic-benchmark` is a toy benchmark for converting synthetic 3D polymer
structures into simulated Hi-C contact matrices and reconstructing 3D
coordinates from those matrices. It exists to evaluate reconstruction methods
against a known ground truth — something real Hi-C data cannot offer.

This is an exploratory research project, **not production code**. Prefer
clarity and reproducibility over premature optimization.

## Architecture

The pipeline is intentionally split into single-responsibility modules so that
each stage can be swapped independently. Data flows in one direction:

```
polymers/  →  forward/  →  reconstruct/  →  eval/
  (truth)    (Hi-C sim)    (inverse)       (compare)
```

- **`polymers/`** — generators that return `(N, 3)` NumPy arrays of bead
  coordinates. Start with random walks and bead-spring chains; polychrom-based
  generators are a future extension.
- **`forward/`** — computes pairwise distances and converts them to contact
  probabilities via `P_ij ∝ d_ij^(-alpha)`, then samples counts. Keep `alpha`,
  noise model, and coverage as explicit parameters.
- **`reconstruct/`** — pluggable backends with a common signature
  `reconstruct(contact_matrix, **kwargs) -> (N, 3) ndarray`. MDS first; later
  add distance-restraint optimization, Pastis, ShRec3D, and eventually neural
  methods.
- **`eval/`** — Procrustes/Kabsch alignment **must** run before any RMSD
  computation. Reconstructions are only defined up to rotation, translation,
  reflection, and global scale.
- **`viz/`** — matplotlib for 2D heatmaps; Plotly for interactive 3D.

## Conventions

- Python 3.12, type hints on all public functions.
- One coordinate convention everywhere: `(N, 3)` arrays, never `(3, N)`.
- One contact-matrix convention: square symmetric `(N, N)`, `dtype=float64` for
  probabilities, `dtype=int64` for sampled counts.
- Every function with stochastic behavior accepts a `seed: int | None = None`
  argument and uses `numpy.random.default_rng(seed)`. **Never** call the global
  `np.random` API.
- Use `hic_benchmark.utils.set_seed` at the top of any notebook or script.
- Configs live in `configs/*.yaml`; never hardcode experiment parameters in
  module code.

## What goes where

- **Module code** (`src/hic_benchmark/`) — reusable, tested, no side effects on
  import.
- **Notebooks** (`notebooks/`) — exploration and figures only. Do not import
  notebook code from modules.
- **Scripts** (`scripts/`, when added) — thin entry points that parse a config
  and call into the package.
- **Tests** (`tests/`) — mirror the `src/` layout. Every public function gets
  at least a smoke test.

## Common commands

```bash
# Activate env
conda activate hic-benchmark

# Run the full test suite
pytest

# Run one test file
pytest tests/test_smoke.py -v

# Lint + format
ruff check .
ruff format .

# Type check
mypy src/

# Pre-commit on all files (run before pushing)
pre-commit run --all-files
```

## Things to avoid

- Do **not** add heavyweight dependencies (PyTorch, OpenMM, polychrom) to the
  core install. They belong behind optional extras when their stages land.
- Do **not** call `np.random.seed` or `random.seed` directly. Use the
  `default_rng` pattern.
- Do **not** compute RMSD without aligning first. This is the most common
  source of meaningless numbers in chromatin reconstruction.
- Do **not** silently densify sparse matrices. Hi-C data is sparse by nature;
  preserve `scipy.sparse` types end-to-end where possible.
- Do **not** commit anything to `data/` or `results/`. They are gitignored for
  a reason.

## When extending

- New polymer model → add a module under `polymers/`, expose it from the
  package `__init__`, add a smoke test that confirms the output shape and
  finite values.
- New reconstruction method → match the existing `reconstruct(...)` signature.
  If it needs extra config, accept a single dict, not a long parameter list.
- New metric → add to `eval/`, ensure it accepts two `(N, 3)` arrays and
  handles the alignment internally so callers can't forget.

## Open questions / known limitations

- The α used in the forward model and the α assumed in the inverse model are
  currently the same. A more honest benchmark would mismatch them deliberately.
- No biological realism yet: no TADs, no loops, no compartments. These come
  with the polychrom integration.
- Single-chromosome only. Whole-genome is explicitly out of scope.
