"""Tests for the PlanGraph -> tensor bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import torch
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_module import (  # noqa: E402
    Connectivity,
    PlanGraph,
    compliance_energy_for_plan,
    plan_to_tensor_repr,
)


def _box(x: float, y: float, w: float, h: float) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _toy_plan() -> PlanGraph:
    """Bedroom -- door -- corridor -- entrance, all dims in meters."""
    bed = _box(0, 0, 4, 4)
    corr = _box(4, 0, 1.4, 6)
    door = _box(3.95, 1.5, 0.9, 0.1)
    entrance = _box(4.6, 5.95, 0.9, 0.1)

    g = nx.Graph()
    g.add_node("bed", geometry=bed, room_type="Bedroom", centroid=(2.0, 2.0))
    g.add_node("corr", geometry=corr, room_type="Corridor", centroid=(4.7, 3.0))
    g.add_edge("bed", "corr", connectivity=Connectivity.DOOR, door_geometry=door)
    g.add_edge(
        "corr",
        "corr",
        connectivity=Connectivity.ENTRANCE,
        door_geometry=entrance,
    )
    return PlanGraph(graph=g, plan_id="toy")


def test_plan_to_tensor_repr_shapes() -> None:
    plan = _toy_plan()
    repr = plan_to_tensor_repr(plan)
    assert repr.door_corners.shape == (2, 4, 2)
    assert repr.corridor_corners.shape == (1, 4, 2)
    assert repr.pairwise_distances.shape == (2, 2)
    assert repr.entrance_mask.shape == (2,)
    assert repr.entrance_mask.sum().item() == 1


def test_plan_to_tensor_repr_distances_consistent() -> None:
    plan = _toy_plan()
    repr = plan_to_tensor_repr(plan)
    assert repr.pairwise_distances[0, 0].item() == 0.0
    assert repr.pairwise_distances[1, 1].item() == 0.0
    assert torch.isfinite(repr.pairwise_distances[0, 1])
    assert repr.pairwise_distances[0, 1].item() > 0.0


def test_compliance_energy_for_plan_compliant_returns_small() -> None:
    plan = _toy_plan()
    e = compliance_energy_for_plan(
        plan,
        door_min_width_m=0.85,
        corridor_min_width_m=1.20,
        egress_max_distance_m=30.0,
    )
    assert e.item() < 1e-2  # only soft-min epsilon


def test_compliance_energy_for_plan_strict_returns_positive() -> None:
    plan = _toy_plan()
    e = compliance_energy_for_plan(
        plan,
        door_min_width_m=1.50,  # impossible
        corridor_min_width_m=1.20,
        egress_max_distance_m=30.0,
    )
    assert e.item() > 0.0
