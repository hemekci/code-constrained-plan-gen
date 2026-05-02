"""Diffusion-backbone interface and a mock implementation.

The interface is intentionally minimal: any backbone that exposes
``predict_x0(x_t, t)`` and ``sample_step(x_t, t)`` can be wrapped by the
``guided_sampling`` loop in ``code_module/guided_sampling.py``.

This decouples our compliance-guided sampling logic from the specific
diffusion backbone (HouseDiffusion, House-GAN++ refactor, or a custom one
trained on MSD), which is useful both for testing and for swapping
backbones across experiments.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

import torch

logger = logging.getLogger(__name__)


class DiffusionBackbone(Protocol):
    """Minimal interface for a diffusion backbone used by the guidance loop."""

    n_steps: int

    def predict_x0(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Tweedie clean-estimate of the data point given the noisy latent."""

    def sample_step(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One unconditional reverse-diffusion step. Returns ``x_{t-1}``."""


@dataclass
class MockDiffusionBackbone:
    """A toy diffusion backbone that "denoises" toward a fixed target plan.

    The forward process adds Gaussian noise to ``target`` according to a
    cosine schedule. The reverse process at step ``t`` returns the analytical
    posterior mean, so unconditional sampling deterministically recovers
    ``target``. This is useful for testing the guidance loop:

    - Without guidance, the loop converges back to ``target`` regardless of
      whether ``target`` is compliant or not.
    - With a compliance-energy guidance term, the trajectory is pulled away
      from ``target`` toward configurations that lower the energy.
    """

    target: torch.Tensor  # the "ground-truth clean" data the backbone wants to recover
    n_steps: int = 50

    def alpha_bar(self, t: int) -> torch.Tensor:
        """Cosine-noise-schedule cumulative alpha at step ``t``."""
        s = 0.008
        f0 = math.cos((s) / (1.0 + s) * math.pi / 2.0) ** 2
        f_t = math.cos(
            ((t / self.n_steps + s) / (1.0 + s)) * math.pi / 2.0
        ) ** 2
        ab = max(f_t / f0, 1e-6)
        return torch.tensor(ab, dtype=self.target.dtype, device=self.target.device)

    def add_noise(self, x0: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        """Forward noising used for tests / scheduling."""
        ab = self.alpha_bar(t)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def predict_x0(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Toy ``x_hat_0`` that depends on ``x_t`` so gradients can flow.

        The contribution of ``x_t`` decays as ``t -> 0``: at full noise
        ``ab=0`` the prediction is a 50/50 mix of target and current latent,
        at ``ab=1`` it is exactly ``target``. This is *not* a faithful
        DDPM x0 predictor, but it is a simple, monotone-toward-target
        function with non-zero ``d predict_x0 / d x_t``, which is all the
        guidance loop needs.
        """
        ab = self.alpha_bar(t)
        weight_xt = (1.0 - ab) * 0.5
        return (1.0 - weight_xt) * self.target + weight_xt * x_t

    def sample_step(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Posterior mean step: blend x_t toward target via the schedule.

        Note for callers: this mock's ``predict_x0`` snaps to ``target`` as
        ``t -> 0`` (a property of any well-behaved DDPM), so the unguided
        endpoint always recovers ``target``. With guidance, the *trajectory*
        is pulled toward lower energy but the endpoint still converges to a
        nearby attractor of the backbone. A real diffusion backbone trained
        on plan data will not collapse to a single fixed target, so guidance
        will accumulate into different endpoints there.
        """
        if t <= 0:
            return self.target.detach().clone()
        ab_t = self.alpha_bar(t)
        ab_prev = self.alpha_bar(t - 1)
        x0_pred = self.predict_x0(x_t, t)
        coef_x0 = ab_prev.sqrt() * (1.0 - ab_t / ab_prev) / (1.0 - ab_t)
        coef_xt = (ab_t / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab_t)
        return coef_x0 * x0_pred + coef_xt * x_t


__all__ = ["DiffusionBackbone", "MockDiffusionBackbone"]
