"""Quick dataset inspection.

Loads a few samples from each dataset and prints schema/shape info.
Saves visualization examples to figures/dataset_samples/ when data is present.

Run after data has been downloaded (see Knowledge/datasets.md).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
FIG_DIR = REPO_ROOT / "figures" / "dataset_samples"


def inspect_msd() -> None:
    """Inspect MSD dataset (multi-apartment building complexes)."""
    msd_dir = DATA_ROOT / "MSD"
    raw_dir = msd_dir / "raw"
    print(f"\n=== MSD ({msd_dir}) ===")
    if not msd_dir.exists():
        print("  ✗ MSD repo not present")
        return
    print(f"  ✓ Repo present: {sorted(p.name for p in msd_dir.iterdir() if not p.name.startswith('.'))[:8]}")
    if not raw_dir.exists():
        print("  ⏳ Raw data not downloaded yet — see Knowledge/datasets.md (Kaggle CLI)")
        return
    files = list(raw_dir.rglob("*"))
    print(f"  ✓ Raw data present: {len(files)} files in {raw_dir}")


def inspect_cubicasa() -> None:
    """Inspect CubiCasa5K dataset (5K Nordic floor plans, SVG vectors)."""
    cc_dir = DATA_ROOT / "CubiCasa5K"
    data_dir = cc_dir / "data"
    print(f"\n=== CubiCasa5K ({cc_dir}) ===")
    if not cc_dir.exists():
        print("  ✗ CubiCasa5K repo not present")
        return
    print(f"  ✓ Repo present: {sorted(p.name for p in cc_dir.iterdir() if not p.name.startswith('.'))[:8]}")
    contents = list(data_dir.glob("*")) if data_dir.exists() else []
    if not contents or all(p.name.startswith(".") for p in contents):
        print("  ⏳ Raw data not downloaded yet — see Knowledge/datasets.md (Zenodo)")
        return
    print(f"  ✓ Raw data present in {data_dir}: {len(contents)} entries")


def inspect_housediffusion() -> None:
    """Inspect HouseDiffusion code repo (baseline reference)."""
    hd_dir = DATA_ROOT / "HouseDiffusion"
    print(f"\n=== HouseDiffusion ({hd_dir}) ===")
    if not hd_dir.exists():
        print("  ✗ HouseDiffusion repo not present")
        return
    pkg_dir = hd_dir / "house_diffusion"
    if pkg_dir.exists():
        py_files = list(pkg_dir.rglob("*.py"))
        print(f"  ✓ Code repo present: {len(py_files)} python files in {pkg_dir.name}/")
    print("  ℹ Code only — no data redistribution; used as baseline reference")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Data root: {DATA_ROOT}")
    inspect_msd()
    inspect_cubicasa()
    inspect_housediffusion()
    print("\nDone. Run after downloading raw data to verify formats.")


if __name__ == "__main__":
    main()
