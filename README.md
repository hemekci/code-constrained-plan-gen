# Regulation-Aware Floor Plan Diffusion

A code-agnostic framework for floor plan generation: diffusion-based generation with training-free
energy guidance and rejection-repair, conditioned on building-code jurisdictions.
**9 jurisdictions × 6 rules** ship out of the box (ISO 21542, EU EN 17210, US ADA+IBC,
UK Approved Doc M+B, DE DIN 18040-2, TR TS 9111, JP Barrier-Free Law, AU AS 1428.1, SG BCA 2019).

## Status
- **Stage**: 1 (Ideation) ✅ + 2 (Development) substantially complete
- **Target venue**: Automation in Construction (IF ~10)
- **Tests**: 26 / 26 passing
- **Real-data baseline**: 380 plans × 9 jurisdictions × 6 rules = 20,520 evaluations across MSD + CubiCasa5K

## Quick Start (Kaggle)

The fastest way to run this end-to-end is the bundled Kaggle smoke notebook.

1. Open a new Kaggle notebook.
2. Right sidebar **Settings** → **Internet: ON**, **Accelerator: NONE** (smoke test) or **GPU T4** (training).
3. Right sidebar **+ Add data** → search `modified-swiss-dwellings` → add the dataset by `caspervanengelenburg`.
4. File → Import notebook → paste the URL of `notebooks/kaggle_smoke.ipynb` from this repo.
5. **Run All**.

Expected output: 26 unit tests pass + a 50-plan compliance baseline matching the local numbers.

## Quick Start (Local)

```bash
git clone https://github.com/hemekci/code-constrained-plan-gen.git
cd code-constrained-plan-gen
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python shapely networkx pytest pandas pyarrow
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch

# Datasets — see data/README.md
.venv/bin/python -m pytest tests/ -v
```

## Knowledge Base
Project notes, literature, plans, experiments, and writing live in the bound Obsidian vault:
- `../obsidian-vault/Code-Constrained-Plan-Gen/`
- Hub: `../obsidian-vault/Code-Constrained-Plan-Gen/00-Hub.md`
- Proposal: `../obsidian-vault/Code-Constrained-Plan-Gen/Plans/research-proposal-v0.md`

## Repo Layout
```
research-code-constrained-plan-gen/
├── data/              # Datasets (RPLAN, CubiCasa5K) — gitignored
├── figures/           # Generated figures for paper
├── output/            # Training outputs, checkpoints — gitignored
├── paper/             # LaTeX manuscript
├── run/conf/          # Hydra configs
├── scripts/           # Standalone scripts
├── src/
│   ├── data_module/
│   ├── model_module/
│   ├── trainer_module/
│   └── utils/
└── tests/
```

## Setup (planned, Stage 2)
```bash
uv venv
uv pip install -e .
```

## License
TBD before release.
