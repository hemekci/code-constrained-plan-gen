"""Verify differentiable energies behave correctly: zero on compliant inputs,
positive on violations, and have gradients pointing in the right direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_module.differentiable import (  # noqa: E402
    corridor_width_energy,
    door_width_energy,
    egress_distance_energy,
    soft_min,
    total_compliance_energy,
)


def _rect_corners(width: float, height: float) -> torch.Tensor:
    """One rectangle as a (4, 2) tensor of corners in CCW order."""
    return torch.tensor(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=torch.float64,
    )


def test_door_energy_is_zero_for_wide_door() -> None:
    door = _rect_corners(0.95, 0.10).unsqueeze(0)  # 1 door, long side 0.95 m
    energy = door_width_energy(door, min_width_m=0.90)
    assert energy.item() < 1e-12


def test_door_energy_is_positive_for_narrow_door() -> None:
    door = _rect_corners(0.70, 0.10).unsqueeze(0)  # 1 door, long side 0.70 m
    energy = door_width_energy(door, min_width_m=0.90)
    assert energy.item() > 0.0
    # Shortfall is 0.20 m → energy = 0.04
    assert abs(energy.item() - 0.04) < 1e-6


def test_door_energy_gradient_points_outward() -> None:
    door = _rect_corners(0.70, 0.10).unsqueeze(0).requires_grad_(True)
    energy = door_width_energy(door, min_width_m=0.90)
    energy.backward()
    grad = door.grad
    assert grad is not None
    # Gradient direction: descending the energy should grow the long side.
    # Corner 0 is at x=0, corner 1 is at x=0.70. Grad on corner 0 x should be
    # positive (energy decreases as we move corner 0 left, i.e., negative grad
    # update would shift it left, growing the side). Verify sign convention:
    long_side_before = torch.linalg.norm(door[0, 0] - door[0, 1]).item()
    step = -0.01 * grad
    moved = (door + step).detach()
    long_side_after = torch.linalg.norm(moved[0, 0] - moved[0, 1]).item()
    assert long_side_after > long_side_before


def test_corridor_energy_is_zero_for_wide_corridor() -> None:
    corr = _rect_corners(6.0, 1.4).unsqueeze(0)  # short side 1.4 m
    energy = corridor_width_energy(corr, min_width_m=1.20)
    assert energy.item() < 1e-12


def test_corridor_energy_is_positive_for_narrow_corridor() -> None:
    corr = _rect_corners(6.0, 0.8).unsqueeze(0)  # short side 0.8 m
    energy = corridor_width_energy(corr, min_width_m=1.20)
    assert energy.item() > 0.0
    # Shortfall is 0.40 m → energy = 0.16
    assert abs(energy.item() - 0.16) < 1e-6


def test_egress_energy_is_zero_when_all_rooms_within_threshold() -> None:
    # 3 rooms; room 2 is the entrance; everyone within 20 m
    distances = torch.tensor(
        [
            [0.0, 5.0, 12.0],
            [5.0, 0.0, 8.0],
            [12.0, 8.0, 0.0],
        ],
        dtype=torch.float64,
    )
    entrance_mask = torch.tensor([False, False, True])
    energy = egress_distance_energy(distances, entrance_mask, max_distance_m=20.0)
    assert energy.item() < 1e-3  # soft-min smoothing introduces tiny epsilon


def test_egress_energy_is_positive_when_far_room_present() -> None:
    distances = torch.tensor(
        [
            [0.0, 80.0, 90.0],
            [80.0, 0.0, 30.0],
            [90.0, 30.0, 0.0],
        ],
        dtype=torch.float64,
    )
    entrance_mask = torch.tensor([False, False, True])
    energy = egress_distance_energy(distances, entrance_mask, max_distance_m=30.0)
    assert energy.item() > 0.0


def test_soft_min_approximates_min() -> None:
    x = torch.tensor([3.0, 7.0, 1.0, 5.0])
    sm = soft_min(x, beta=20.0).item()
    assert abs(sm - 1.0) < 0.05


def test_total_energy_runs_end_to_end() -> None:
    # 1 narrow door, 1 narrow corridor, 1 far room
    door = _rect_corners(0.70, 0.10).unsqueeze(0)
    corr = _rect_corners(6.0, 0.8).unsqueeze(0)
    distances = torch.tensor(
        [
            [0.0, 80.0],
            [80.0, 0.0],
        ],
        dtype=torch.float64,
    )
    entrance_mask = torch.tensor([False, True])
    e = total_compliance_energy(
        door_corners=door,
        corridor_corners=corr,
        pairwise_distances=distances,
        entrance_mask=entrance_mask,
        door_min_width_m=0.90,
        corridor_min_width_m=1.20,
        egress_max_distance_m=30.0,
    )
    assert e.item() > 0.0


def test_total_energy_is_zero_when_compliant() -> None:
    door = _rect_corners(0.95, 0.10).unsqueeze(0)
    corr = _rect_corners(6.0, 1.4).unsqueeze(0)
    distances = torch.tensor(
        [
            [0.0, 5.0],
            [5.0, 0.0],
        ],
        dtype=torch.float64,
    )
    entrance_mask = torch.tensor([False, True])
    e = total_compliance_energy(
        door_corners=door,
        corridor_corners=corr,
        pairwise_distances=distances,
        entrance_mask=entrance_mask,
        door_min_width_m=0.90,
        corridor_min_width_m=1.20,
        egress_max_distance_m=30.0,
    )
    assert e.item() < 1e-2  # soft-min epsilon


def test_total_energy_supports_autograd() -> None:
    door = _rect_corners(0.70, 0.10).unsqueeze(0).requires_grad_(True)
    corr = _rect_corners(6.0, 0.8).unsqueeze(0).requires_grad_(True)
    distances = torch.tensor(
        [[0.0, 80.0], [80.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    entrance_mask = torch.tensor([False, True])
    e = total_compliance_energy(
        door_corners=door,
        corridor_corners=corr,
        pairwise_distances=distances,
        entrance_mask=entrance_mask,
        door_min_width_m=0.90,
        corridor_min_width_m=1.20,
        egress_max_distance_m=30.0,
    )
    e.backward()
    assert door.grad is not None and door.grad.abs().sum() > 0
    assert corr.grad is not None and corr.grad.abs().sum() > 0
    assert distances.grad is not None and distances.grad.abs().sum() > 0
