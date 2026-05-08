# hic-benchmark

A toy benchmark for converting synthetic 3D polymer structures into Hi-C contact
matrices and reconstructing 3D coordinates.

## What this project does

The reconstruction of chromatin 3D structure from Hi-C contact data is an
ill-posed inverse problem, and real Hi-C datasets rarely come with a known
ground-truth conformation to validate against. `hic-benchmark` sidesteps that
by working in a fully synthetic regime:

```
synthetic 3D polymer
        │
        ▼
pairwise distance matrix
        │
        ▼
simulated Hi-C contact matrix
        │
        ▼
contact-to-distance transformation
        │
        ▼
3D reconstruction (MDS, ...)
        │
        ▼
structure comparison vs. ground truth
```

Because the original structure is known, reconstruction accuracy can be
measured directly under varying noise, sparsity, and structural complexity.

## Status

Early scaffolding. See the **Roadmap** below for what's implemented vs. planned.

## Installation

Requires [Miniforge](https://github.com/conda-forge/miniforge) (or any conda
distribution defaulting to the `conda-forge` channel).

```bash
git clone <your-repo-url> hic-benchmark
cd hic-benchmark
conda env create -f environment.yml
conda activate hic-benchmark
pip install -e .
pre-commit install
```

Verify the install:

```bash
pytest
```

## Quick start

```python
from hic_benchmark.polymers import random_walk
from hic_benchmark.forward import distances_to_contacts
from hic_benchmark.reconstruct import mds_reconstruct
from hic_benchmark.eval import procrustes_rmsd

coords = random_walk(n_beads=100, seed=0)
contacts = distances_to_contacts(coords, alpha=1.0)
recovered = mds_reconstruct(contacts, alpha=1.0)
print("RMSD:", procrustes_rmsd(coords, recovered))
```

## Project structure

```
hic-benchmark/
├── src/hic_benchmark/
│   ├── polymers/      # synthetic 3D structure generators
│   ├── forward/       # structure → distances → contact matrix
│   ├── reconstruct/   # contact matrix → 3D coordinates
│   ├── eval/          # alignment + reconstruction metrics
│   ├── viz/           # plotting helpers
│   └── utils.py       # seeding, shared helpers
├── tests/
├── configs/           # YAML experiment configs (added later)
├── data/              # generated structures + .cool files (gitignored)
├── results/           # metrics and figures (gitignored)
├── notebooks/         # exploratory work only
├── environment.yml
└── pyproject.toml
```

## Roadmap

- [x] Project scaffolding
- [ ] Random-walk and bead-spring polymer generators
- [ ] Forward model: distances → contact probabilities → sampled counts
- [ ] `.cool` file I/O via `cooler`
- [ ] MDS-based reconstruction
- [ ] Procrustes alignment + RMSD / correlation metrics
- [ ] Parameter sweeps (noise, sparsity, α)
- [ ] Polychrom-based realistic polymers (TADs, loops)
- [ ] Neural reconstruction (GNN / diffusion)

## License

TBD.
