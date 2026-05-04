"""HouseDiffusion backbone adapter.

Wraps the upstream HouseDiffusion (Shabani et al., CVPR 2023) sampler to
conform to the local ``DiffusionBackbone`` protocol so that
``universal_guidance_sample`` can drive it without modification.

This module is *deliberately* import-light: HouseDiffusion has heavy pinned
dependencies (TF 2.11, mpi4py, an old shapely / networkx pair) and we do not
want them to leak into the main project venv. We resolve the upstream package
lazily on first use, which lets the protocol-only tests in this repo run with
just torch.

Install path:
    bash scripts/install_housediffusion.sh
which sets up a forked uv environment under ``data/HouseDiffusion/.venv-hd``
and exposes the upstream ``house_diffusion`` Python package on its own sys.path
when this adapter is invoked from that env.

Design notes
------------
* HouseDiffusion samples an (N, 32, 2) tensor of room corners on a Gaussian
  schedule. We expose that same shape unchanged so the energy fn (which is
  shape-agnostic) sees real plan coordinates.
* ``predict_x0`` is implemented via Tweedie's identity using the model's
  predicted noise: ``x_hat_0 = (x_t - sqrt(1-ab) * eps) / sqrt(ab)``.
* ``sample_step`` reuses HouseDiffusion's own posterior-mean step so that the
  unconditional trajectory is unchanged when the guidance term is zero.

This file does **not** re-implement HouseDiffusion. It only adapts it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - typing only
    HDModel = Any
    HDDiffusion = Any


def _lazy_import_housediffusion() -> tuple[Any, Any]:
    """Import upstream ``house_diffusion`` lazily.

    Raises a clear error pointing at the install script if the package is not
    importable in the current interpreter (the expected case in the main
    project venv).
    """
    try:
        from house_diffusion import gaussian_diffusion as gd  # type: ignore
        from house_diffusion.script_util import (  # type: ignore
            create_model_and_diffusion,
        )
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "house_diffusion is not installed in this interpreter. "
            "Run scripts/install_housediffusion.sh and re-run from the "
            "forked env (data/HouseDiffusion/.venv-hd)."
        ) from exc
    return gd, create_model_and_diffusion


@dataclass
class HouseDiffusionBackbone:
    """Adapter conforming to ``DiffusionBackbone`` over an upstream HD model.

    Parameters
    ----------
    model
        An instantiated upstream HouseDiffusion U-Net-style network.
    diffusion
        The upstream ``GaussianDiffusion`` instance with its noise schedule.
    model_kwargs
        Extra conditioning passed to the model's ``forward`` (room types,
        graph encoding, etc.). Held constant across the sampling loop.
    """

    model: Any
    diffusion: Any
    model_kwargs: Optional[dict[str, Any]] = None

    @property
    def n_steps(self) -> int:
        return int(self.diffusion.num_timesteps)

    def _alpha_bar(self, t: int) -> torch.Tensor:
        ab = float(self.diffusion.alphas_cumprod[t])
        return torch.tensor(ab, dtype=torch.float64)

    def predict_x0(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Tweedie clean estimate from upstream eps prediction."""
        kwargs = self.model_kwargs or {}
        t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        eps = self.model(x_t, t_tensor, **kwargs)
        ab = self._alpha_bar(t).to(x_t)
        return (x_t - (1.0 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-6)

    def sample_step(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Single reverse step delegated to upstream posterior-mean sampler."""
        kwargs = self.model_kwargs or {}
        t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        out = self.diffusion.p_sample(
            self.model,
            x_t,
            t_tensor,
            clip_denoised=False,
            model_kwargs=kwargs,
        )
        return out["sample"]


def load_housediffusion_backbone(
    checkpoint_path: str,
    *,
    model_kwargs: Optional[dict[str, Any]] = None,
    device: str = "cpu",
) -> HouseDiffusionBackbone:
    """Construct an adapter from a checkpoint path.

    Defers all heavy imports until called so that test-only code paths in the
    main venv stay fast.
    """
    _, create_model_and_diffusion = _lazy_import_housediffusion()
    model, diffusion = create_model_and_diffusion()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return HouseDiffusionBackbone(
        model=model, diffusion=diffusion, model_kwargs=model_kwargs
    )


__all__ = ["HouseDiffusionBackbone", "load_housediffusion_backbone"]
