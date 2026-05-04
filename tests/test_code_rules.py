"""Synthetic-plan tests for the code-rule module.

Verifies the rule-checker API works on hand-crafted PlanGraph instances
without needing the MSD raw data.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import pytest
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_module import (  # noqa: E402
    Connectivity,
    CorridorMinWidth,
    DeadEndCorridor,
    DoorWidth,
    EgressTravelDistance,
    PlanGraph,
    evaluate_jurisdiction,
    get_jurisdiction,
)


def _box(x: float, y: float, w: float, h: float) -> Polygon:
    """Axis-aligned rectangle at (x, y) with width w and height h."""
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _door(x: float, y: float, w: float, h: float = 0.05) -> Polygon:
    """Tiny rectangle representing a door opening; min dimension is w (or h)."""
    return _box(x, y, w, h)


def _make_compliant_plan() -> PlanGraph:
    """Two bedrooms + corridor + entrance, all dimensions in meters.

    Layout (rough):
       BED1 (4x4) -- door(0.9) -- CORR (1.4x6) -- door(0.9) -- BED2 (4x4)
                                       |
                                  entrance
    """
    bed1 = _box(0, 0, 4, 4)
    bed2 = _box(5.4, 0, 4, 4)
    corr = _box(4, 0, 1.4, 6)  # 1.4 m wide corridor

    g = nx.Graph()
    g.add_node("bed1", geometry=bed1, room_type="Bedroom", centroid=(2.0, 2.0))
    g.add_node("bed2", geometry=bed2, room_type="Bedroom", centroid=(7.4, 2.0))
    g.add_node("corr", geometry=corr, room_type="Corridor", centroid=(4.7, 3.0))

    door1 = _door(3.95, 1.5, 0.9, 0.1)  # bed1 -- corr
    door2 = _door(5.35, 1.5, 0.9, 0.1)  # corr -- bed2
    entrance_door = _door(4.6, 5.95, 0.9, 0.1)  # corr -- outside

    g.add_edge("bed1", "corr", connectivity=Connectivity.DOOR, door_geometry=door1)
    g.add_edge("corr", "bed2", connectivity=Connectivity.DOOR, door_geometry=door2)
    # Self-loop with entrance connectivity so corridor is reachable from outside
    g.add_edge(
        "corr",
        "corr",
        connectivity=Connectivity.ENTRANCE,
        door_geometry=entrance_door,
    )

    return PlanGraph(graph=g, unit_scale_m_per_unit=1.0, plan_id="test-compliant")


def _make_narrow_corridor_plan() -> PlanGraph:
    """Same layout but corridor width 0.6 m — violates min 1.20 m."""
    bed1 = _box(0, 0, 4, 4)
    bed2 = _box(4.6, 0, 4, 4)
    corr = _box(4, 0, 0.6, 6)  # narrow!

    g = nx.Graph()
    g.add_node("bed1", geometry=bed1, room_type="Bedroom", centroid=(2.0, 2.0))
    g.add_node("bed2", geometry=bed2, room_type="Bedroom", centroid=(6.6, 2.0))
    g.add_node("corr", geometry=corr, room_type="Corridor", centroid=(4.3, 3.0))

    door1 = _door(3.95, 1.5, 0.9, 0.1)
    door2 = _door(4.55, 1.5, 0.9, 0.1)
    entrance_door = _door(4.0, 5.95, 0.9, 0.1)

    g.add_edge("bed1", "corr", connectivity=Connectivity.DOOR, door_geometry=door1)
    g.add_edge("corr", "bed2", connectivity=Connectivity.DOOR, door_geometry=door2)
    g.add_edge(
        "corr",
        "corr",
        connectivity=Connectivity.ENTRANCE,
        door_geometry=entrance_door,
    )
    return PlanGraph(graph=g, plan_id="test-narrow")


def _make_narrow_door_plan() -> PlanGraph:
    """Compliant corridor; one door is only 0.65 m wide."""
    plan = _make_compliant_plan()
    g = plan.graph.copy()
    narrow_door = _door(3.95, 1.5, 0.65, 0.1)
    g["bed1"]["corr"]["door_geometry"] = narrow_door
    return PlanGraph(graph=g, plan_id="test-narrow-door")


def _make_far_room_plan() -> PlanGraph:
    """Bedroom 1 placed 50 m from the corridor, exceeding 30 m egress limit."""
    plan = _make_compliant_plan()
    g = plan.graph.copy()
    g.nodes["bed1"]["centroid"] = (-50.0, 2.0)
    return PlanGraph(graph=g, plan_id="test-far-room")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compliant_plan_passes_all_rules() -> None:
    plan = _make_compliant_plan()
    assert DoorWidth(min_width_m=0.85).check(plan).passed
    assert CorridorMinWidth(min_width_m=1.20).check(plan).passed
    assert EgressTravelDistance(max_distance_m=30.0).check(plan).passed
    assert DeadEndCorridor(max_dead_end_length_m=6.5).check(plan).passed


def test_narrow_corridor_fails_corridor_rule() -> None:
    plan = _make_narrow_corridor_plan()
    result = CorridorMinWidth(min_width_m=1.20).check(plan)
    assert not result.passed
    assert result.score < 1.0
    assert len(result.details["violations"]) == 1
    violation = result.details["violations"][0]
    assert violation["width_m"] < 1.20


def test_narrow_door_fails_door_rule_and_has_positive_energy() -> None:
    plan = _make_narrow_door_plan()
    rule = DoorWidth(min_width_m=0.80)
    result = rule.check(plan)
    assert not result.passed
    energy = rule.energy(plan)
    assert energy > 0.0
    # Compliant plan has zero energy
    assert rule.energy(_make_compliant_plan()) == pytest.approx(0.0)


def test_far_room_fails_egress_rule() -> None:
    plan = _make_far_room_plan()
    result = EgressTravelDistance(max_distance_m=30.0).check(plan)
    assert not result.passed
    distances = result.details["per_room_distance_m"]
    assert distances["bed1"] > 30.0


def test_jurisdiction_TR_evaluates_all_rules() -> None:
    plan = _make_compliant_plan()
    report = evaluate_jurisdiction(plan, "TR")
    assert report["jurisdiction"] == "TR"
    # 6 rules: door_width, corridor_min_width, egress_travel_distance,
    # dead_end_corridor, at_least_n_exits, wheelchair_turn_radius
    assert report["aggregate"]["n_total"] == 6
    assert math.isfinite(report["aggregate"]["compliance_score"])


def test_jurisdiction_specs_have_distinct_thresholds() -> None:
    tr = dict(get_jurisdiction("TR").rules)
    us = dict(get_jurisdiction("US").rules)
    iso = dict(get_jurisdiction("ISO").rules)
    assert tr["door_width"]["min_width_m"] != us["door_width"]["min_width_m"]
    assert (
        tr["egress_travel_distance"]["max_distance_m"]
        != us["egress_travel_distance"]["max_distance_m"]
    )
    assert iso["dead_end_corridor"]["max_dead_end_length_m"] > 0.0


def test_global_jurisdiction_set_is_complete() -> None:
    """Sanity check that all 9 representative jurisdictions are registered."""
    from code_module import JURISDICTIONS

    expected = {"ISO", "EU", "US", "UK", "DE", "TR", "JP", "AU", "SG"}
    assert expected <= set(JURISDICTIONS), (
        f"Missing jurisdictions: {expected - set(JURISDICTIONS)}"
    )
    for code, spec in JURISDICTIONS.items():
        assert spec.build(), f"Jurisdiction {code} produced empty rule list"


def _make_plan_with_narrow_bedroom_door() -> PlanGraph:
    """Compliant accessible-route doors; one bedroom-bedroom door is narrow."""
    bed1 = _box(0, 0, 4, 4)
    bed2 = _box(5.4, 0, 4, 4)
    bed3 = _box(10.8, 0, 4, 4)
    corr = _box(4, 0, 1.4, 6)
    g = nx.Graph()
    g.add_node("bed1", geometry=bed1, room_type="Bedroom", centroid=(2.0, 2.0))
    g.add_node("bed2", geometry=bed2, room_type="Bedroom", centroid=(7.4, 2.0))
    g.add_node("bed3", geometry=bed3, room_type="Bedroom", centroid=(12.8, 2.0))
    g.add_node("corr", geometry=corr, room_type="Corridor", centroid=(4.7, 3.0))
    # Accessible-route doors (corridor-bedroom): wide
    g.add_edge(
        "bed1", "corr",
        connectivity=Connectivity.DOOR,
        door_geometry=_door(3.95, 1.5, 0.95, 0.1),
    )
    g.add_edge(
        "corr", "bed2",
        connectivity=Connectivity.DOOR,
        door_geometry=_door(5.35, 1.5, 0.95, 0.1),
    )
    # Bedroom-bedroom internal door: narrow (bypassed by accessible_route scope)
    g.add_edge(
        "bed2", "bed3",
        connectivity=Connectivity.DOOR,
        door_geometry=_door(9.35, 1.5, 0.65, 0.1),
    )
    g.add_edge(
        "corr", "corr",
        connectivity=Connectivity.ENTRANCE,
        door_geometry=_door(4.6, 5.95, 0.95, 0.1),
    )
    return PlanGraph(graph=g, plan_id="test-permissive")


def test_door_scope_accessible_route_skips_bedroom_internal_door() -> None:
    """The permissive scope must ignore non-accessible-route doors."""
    plan = _make_plan_with_narrow_bedroom_door()
    strict = DoorWidth(min_width_m=0.90, scope="all").check(plan)
    permissive = DoorWidth(min_width_m=0.90, scope="accessible_route").check(plan)
    assert not strict.passed, "strict scope must catch the narrow bedroom door"
    assert permissive.passed, (
        "accessible_route scope must skip the bedroom-bedroom door "
        f"(violations={permissive.details['violations']})"
    )
    # n_doors counts also reflect the scope: permissive sees fewer doors
    assert permissive.details["n_doors"] < strict.details["n_doors"]


def test_permissive_door_jurisdiction_matches_underlying_thresholds() -> None:
    from code_module import permissive_door_jurisdiction

    base = get_jurisdiction("TR")
    perm = permissive_door_jurisdiction("TR")
    base_door = dict(base.rules)["door_width"]
    perm_door = dict(perm.rules)["door_width"]
    assert perm_door["min_width_m"] == base_door["min_width_m"]
    assert perm_door["scope"] == "accessible_route"
    # Egress rule is unchanged
    assert dict(perm.rules)["egress_travel_distance"] == dict(base.rules)[
        "egress_travel_distance"
    ]
