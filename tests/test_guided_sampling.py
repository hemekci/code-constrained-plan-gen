"""Verify the guided-sampling loop reduces compliance energy when given a
non-compliant target.

Setup: the mock backbone wants to recover a 'narrow door' target. Without
guidance, the loop converges to that narrow-door target. With guidance based
on door-width energy, the loop should pull the trajectory toward a wider
door.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_module.differentiable import door_width_energy  # noqa: E402
from code_module.guided_sampling import universal_guidance_sample  # noqa: E402
from model_module import MockDiffusionBackbone  # noqa: E402


def _door_corners(width: float, height: float = 0.10) -> torch.Tensor:
    """One door as a (1, 4, 2) tensor — long side = ``width`` m."""
    return torch.tensor(
        [[[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]],
        dtype=torch.float64,
    )


def _energy_with_threshold(min_width_m: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return door_width_energy(x, min_width_m=min_width_m)
    return fn


def test_unguided_recovery_yields_narrow_door() -> None:
    target = _door_corners(0.70)
    backbone = MockDiffusionBackbone(target=target, n_steps=20)
    x_T = torch.randn_like(target)
    x_0, history = universal_guidance_sample(
        backbone,
        x_T,
        energy_fn=_energy_with_threshold(0.90),
        guidance_scale=0.0,  # no guidance
    )
    long_side_recovered = (x_0[0, 0] - x_0[0, 1]).norm().item()
    long_side_target = (target[0, 0] - target[0, 1]).norm().item()
    assert abs(long_side_recovered - long_side_target) < 1e-3
    # Without guidance, ending energy equals shapeful door-width shortfall
    assert history.energies[-1] > 0.0


def test_guided_sampling_reduces_compliance_energy() -> None:
    target = _door_corners(0.70)  # backbone wants a 0.70 m door
    backbone = MockDiffusionBackbone(target=target, n_steps=30)
    x_T = torch.randn_like(target)
    energy_fn = _energy_with_threshold(0.90)

    _, hist_unguided = universal_guidance_sample(
        backbone, x_T, energy_fn=energy_fn, guidance_scale=0.0
    )
    _, hist_guided = universal_guidance_sample(
        backbone, x_T, energy_fn=energy_fn, guidance_scale=0.3
    )

    # First-step energies should match (same x_T); final energies should differ
    assert abs(hist_unguided.energies[0] - hist_guided.energies[0]) < 1e-6
    final_unguided = hist_unguided.energies[-1]
    final_guided = hist_guided.energies[-1]
    assert final_guided < final_unguided


def test_guidance_grad_norms_are_nonzero_in_violation_regime() -> None:
    target = _door_corners(0.70)
    backbone = MockDiffusionBackbone(target=target, n_steps=10)
    x_T = torch.randn_like(target)
    _, history = universal_guidance_sample(
        backbone,
        x_T,
        energy_fn=_energy_with_threshold(0.90),
        guidance_scale=0.1,
    )
    assert any(g > 0.0 for g in history.grad_norms)


def test_history_records_all_steps_at_log_every_1() -> None:
    target = _door_corners(0.95)  # already compliant — energy ~ 0
    backbone = MockDiffusionBackbone(target=target, n_steps=12)
    x_T = torch.randn_like(target)
    _, history = universal_guidance_sample(
        backbone,
        x_T,
        energy_fn=_energy_with_threshold(0.90),
        guidance_scale=0.1,
        log_every=1,
    )
    assert len(history.steps) == 12
    assert len(history.energies) == 12
    assert len(history.grad_norms) == 12
