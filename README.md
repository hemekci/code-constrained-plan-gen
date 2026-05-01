# Code-Constrained Floor Plan Generation

Building-code-constrained floor plan generation: A diffusion approach with hard constraints on accessibility and fire egress.

## Status
- **Stage**: Ideation (Stage 1)
- **Target venue**: Automation in Construction (IF ~10)
- **Created**: 2026-05-02

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
