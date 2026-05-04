"""Tests for the guided-sampling driver, especially the pre-flight gate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_guided_sampling as rgs  # noqa: E402


def test_preflight_passes_on_mock_backbone_shape() -> None:
    """The (B, 4, 2) shape used by MockDiffusionBackbone must validate cleanly."""
    assert rgs.preflight_validate_shape("TR", (1, 4, 2)) is True


def test_preflight_passes_on_housediffusion_shape_with_adapter() -> None:
    """(B, N_rooms, 32, 2) must validate when the housediffusion adapter is on."""
    assert rgs.preflight_validate_shape(
        "TR", (1, 1, 32, 2), shape_adapter="housediffusion"
    ) is True


def test_preflight_fails_on_housediffusion_shape_without_adapter() -> None:
    """Without the adapter, (B, N_rooms, 32, 2) silently miswires; must be flagged."""
    assert rgs.preflight_validate_shape(
        "TR", (1, 1, 32, 2), shape_adapter="identity"
    ) is False


def test_preflight_rejects_obviously_wrong_rank() -> None:
    """A 1-D tensor cannot encode (..., V, 2) and must be rejected."""
    assert rgs.preflight_validate_shape("TR", (8,)) is False


def test_pilot_verdict_blocks_when_no_improvement() -> None:
    """Pilot verdict must say NO-GO when guidance does not reduce energy."""
    bad_results = [
        rgs.PerPlanResult(
            plan_idx=i,
            unguided_final_energy=1.0,
            guided_final_energy=1.1,  # WORSE under guidance
            delta_energy=-0.1,
            guided_grad_norm_max=0.5,
        )
        for i in range(8)
    ]
    verdict = rgs.evaluate_pilot(bad_results)
    assert verdict.go is False
    assert verdict.fraction_improved == 0.0


def test_pilot_verdict_passes_when_majority_improves() -> None:
    """6/8 improved with positive median -> GO."""
    mixed = []
    for i in range(8):
        # 6 improvements, 2 regressions
        delta = +0.5 if i < 6 else -0.05
        mixed.append(
            rgs.PerPlanResult(
                plan_idx=i,
                unguided_final_energy=1.0,
                guided_final_energy=1.0 - delta,
                delta_energy=delta,
                guided_grad_norm_max=0.5,
            )
        )
    verdict = rgs.evaluate_pilot(mixed)
    assert verdict.go is True
    assert verdict.fraction_improved == 0.75
