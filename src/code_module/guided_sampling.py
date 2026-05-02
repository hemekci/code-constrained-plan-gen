"""Universal-Guidance-style compliance-guided diffusion sampling.

Pseudocode (Bansal et al. 2023 forward universal guidance):

    for t in T..0:
        x_hat_0 = backbone.predict_x0(x_t, t)               # Tweedie estimate
        energy  = compliance_energy(x_hat_0; jurisdiction)  # differentiable
        grad    = autograd(energy, x_t)                     # via x_t -> x_hat_0
        x_t     = x_t - guidance_scale * grad
        x_{t-1} = backbone.sample_step(x_t, t)

This file implements that loop in a backbone-agnostic way; the energy
function is a callable producing a scalar tensor from a predicted clean
data point ``x_hat_0`` of arbitrary shape.

For the integration with our compliance rules, callers will typically pass
a closure that wraps a tensor-shaped plan into the interpretation expected
by ``total_compliance_energy``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


EnergyFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class GuidedSamplingHistory:
    """Per-step trajectory bookkeeping for analysis and tests."""

    steps: list[int] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)


def universal_guidance_sample(
    backbone,
    x_T: torch.Tensor,
    energy_fn: EnergyFn,
    *,
    guidance_scale: float = 1.0,
    log_every: int = 1,
) -> tuple[torch.Tensor, GuidedSamplingHistory]:
    """Run guided reverse-diffusion sampling.

    Parameters
    ----------
    backbone
        Anything implementing the ``DiffusionBackbone`` protocol
        (``n_steps``, ``predict_x0(x_t, t)``, ``sample_step(x_t, t)``).
    x_T
        Initial noisy latent at the largest timestep.
    energy_fn
        Differentiable scalar energy on ``x_hat_0``. Lower is more compliant.
    guidance_scale
        Step size for the guidance gradient. Bansal et al. tune this per
        modality; small enough to avoid driving x_t off-manifold, large
        enough to actually steer.

    Returns
    -------
    Final ``x_0`` estimate and a ``GuidedSamplingHistory`` with per-step
    energy and gradient norms (useful for ablation plots).
    """
    history = GuidedSamplingHistory()
    x_t = x_T.detach().clone()
    for t in range(backbone.n_steps, 0, -1):
        # Guidance: enable autograd through predict_x0 by branching on x_t
        x_t_grad = x_t.detach().clone().requires_grad_(True)
        x_hat_0 = backbone.predict_x0(x_t_grad, t)
        energy = energy_fn(x_hat_0)
        if energy.requires_grad:
            grad = torch.autograd.grad(energy, x_t_grad, allow_unused=True)[0]
        else:
            grad = None

        if grad is None:
            grad_norm = 0.0
        else:
            grad_norm = float(grad.norm().item())
            x_t = x_t - guidance_scale * grad

        x_t = backbone.sample_step(x_t, t)

        if (t % log_every) == 0:
            history.steps.append(t)
            history.energies.append(float(energy.detach().item()))
            history.grad_norms.append(grad_norm)

    return x_t, history


__all__ = ["EnergyFn", "GuidedSamplingHistory", "universal_guidance_sample"]
