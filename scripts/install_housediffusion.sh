#!/usr/bin/env bash
# Install upstream HouseDiffusion (Shabani et al., CVPR 2023) into a *forked*
# uv venv so its heavy pinned deps (TF 2.11, mpi4py, old shapely/networkx) do
# not pollute the main project venv.
#
# Usage:
#   bash scripts/install_housediffusion.sh
#
# After install:
#   source data/HouseDiffusion/.venv-hd/bin/activate
#   python -c "import house_diffusion; print(house_diffusion.__file__)"
#
# The HouseDiffusion repo is expected to already be cloned at
# data/HouseDiffusion (see data/README.md).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HD_DIR="${REPO_ROOT}/data/HouseDiffusion"
VENV_DIR="${HD_DIR}/.venv-hd"

if [ ! -d "${HD_DIR}" ]; then
  echo "data/HouseDiffusion is missing. Clone it first:"
  echo "  git clone https://github.com/aminshabani/house_diffusion.git ${HD_DIR}"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH. Install it from https://docs.astral.sh/uv/ first."
  exit 1
fi

echo "[install_housediffusion] Creating venv at ${VENV_DIR} (Python 3.10 — TF 2.11 requires <3.11)"
uv venv --python 3.10 "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[install_housediffusion] Installing pinned deps from upstream + this repo"
uv pip install --upgrade pip wheel
# Upstream deps. mpi4py is replaced by a no-op shim later if it fails to build.
uv pip install \
  "torch==2.0.1" \
  "torchvision==0.15.2" \
  "tensorflow==2.11.0" \
  "numpy<2" \
  "shapely>=2.0" \
  "networkx>=3.0" \
  "pandas" \
  "pytest"

# Upstream package — installed editable so that any local hotfixes are picked up.
echo "[install_housediffusion] pip install -e ${HD_DIR}"
uv pip install -e "${HD_DIR}" || {
  echo "[install_housediffusion] Editable install failed. Falling back to PYTHONPATH."
  echo "Add this to your shell when running training:"
  echo "  export PYTHONPATH=${HD_DIR}:\$PYTHONPATH"
}

echo "[install_housediffusion] Done. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
