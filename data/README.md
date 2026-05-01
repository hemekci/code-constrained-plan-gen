# `data/` — Datasets and external repos

This directory holds large external datasets and reference code repos. **Nothing here is committed** except this README and `annotations/` (our own contribution).

See `../obsidian-vault/Code-Constrained-Plan-Gen/Knowledge/datasets.md` for the canonical access plan.

## Expected Layout (after setup)

```
data/
├── README.md                  # this file (committed)
├── annotations/               # our code-compliance annotations (committed)
│   └── code-compliance/
│       ├── msd/
│       └── cubicasa5k/
│
├── MSD/                       # ⬇ git clone https://github.com/caspervanengelenburg/msd.git
│   └── raw/                   # ⬇ kaggle datasets download -d caspervanengelenburg/modified-swiss-dwellings
│
├── CubiCasa5K/                # ⬇ git clone https://github.com/CubiCasa/CubiCasa5k.git
│   └── data/                  # ⬇ wget https://zenodo.org/record/2613548/files/cubicasa5k.zip
│
├── HouseDiffusion/            # ⬇ git clone https://github.com/aminshabani/house_diffusion.git
│
└── cache/                     # parquet/lmdb caches (regenerable)
```

## Setup Commands (TL;DR)

```bash
cd data

# 1. Code repos (cheap)
git clone --depth 1 https://github.com/caspervanengelenburg/msd.git MSD
git clone --depth 1 https://github.com/CubiCasa/CubiCasa5k.git CubiCasa5K
git clone --depth 1 https://github.com/aminshabani/house_diffusion.git HouseDiffusion

# 2. MSD raw data (Kaggle, requires API key in ~/.kaggle/kaggle.json)
pip install kaggle
mkdir -p MSD/raw
kaggle datasets download -d caspervanengelenburg/modified-swiss-dwellings -p MSD/raw --unzip

# 3. CubiCasa5K raw data (Zenodo, ~3 GB)
cd CubiCasa5K/data
wget https://zenodo.org/record/2613548/files/cubicasa5k.zip
unzip cubicasa5k.zip
cd ../..
```

## Verify
```bash
python ../scripts/inspect_datasets.py
```
