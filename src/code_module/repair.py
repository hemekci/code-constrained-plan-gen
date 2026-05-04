"""Hard rejection-repair routines for non-differentiable rules.

These run on a final ``x_hat_0`` (or any decoded ``PlanGraph``) after the
guidance loop terminates. They are deliberately conservative: each repair
either applies a small, locally-scoped change to make the offending rule
pass, or it returns the plan unchanged and reports the failure so that the
sampler can resample if needed.

Pipeline contract:

    1. Run guided sampling (soft compliance energies).
    2. Decode the final tensor into a ``PlanGraph``.
    3. Pass it to ``apply_hard_repairs(plan, jurisdiction_code)`` to handle
       the three discrete rules: dead_end_corridor, at_least_n_exits,
       wheelchair_turn_radius.
    4. Re-run ``evaluate_jurisdiction`` to confirm overall compliance.

The repair functions never mutate the input ``PlanGraph``; they return a
new one with the modified ``networkx.Graph``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import networkx as nx
from shapely.geometry import Polygon

from .accessibility import (
    _polygon_inscribed_circle_diameter,
    _polygon_max_dimension,
    _polygon_min_dimension,
)
from .base import Connectivity, PlanGraph
from .jurisdictions import get_jurisdiction

logger = logging.getLogger(__name__)


@dataclass
class RepairReport:
    """Per-rule outcome of a repair pass."""

    repaired: list[str] = field(default_factory=list)
    unrepairable: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        return not self.unrepairable


def _scale(plan: PlanGraph) -> float:
    return float(plan.unit_scale_m_per_unit)


def _corridor_nodes(plan: PlanGraph) -> list[Any]:
    return [
        nid
        for nid, room in plan.rooms().items()
        if room.room_type == "Corridor"
    ]


# ---------------------------------------------------------------------------
# at_least_n_exits — promote the boundary-closest room to an entrance
# ---------------------------------------------------------------------------


def _component_subgraphs(plan: PlanGraph) -> list[set[Any]]:
    rooms = plan.rooms()
    g = nx.Graph()
    g.add_nodes_from(rooms.keys())
    for u, v, _ in plan.graph.edges(data=True):
        if u in rooms and v in rooms and u != v:
            g.add_edge(u, v)
    return [set(c) for c in nx.connected_components(g)]


def _plan_bbox(plan: PlanGraph) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for room in plan.rooms().values():
        x0, y0, x1, y1 = room.geometry.bounds
        xs.extend([x0, x1])
        ys.extend([y0, y1])
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def repair_at_least_n_exits(
    plan: PlanGraph, *, min_exits: int = 1
) -> tuple[PlanGraph, RepairReport]:
    """Ensure every connected component has at least ``min_exits`` entrance edges.

    Strategy: for each under-served component, pick the room closest to the
    plan bbox boundary (the room most plausibly on the building perimeter)
    and add an Entrance self-loop. The added entrance carries no door
    geometry — downstream rules that require door geometry should treat it
    as a synthetic exit.
    """
    report = RepairReport()
    rooms = plan.rooms()
    if not rooms:
        report.notes.append("plan has no rooms; nothing to repair")
        return plan, report

    bbox = _plan_bbox(plan)
    g = plan.graph.copy()
    entrance_set = set(plan.entrance_nodes())

    for comp in _component_subgraphs(plan):
        comp_rooms = [r for r in comp if r in rooms]
        n_exits = sum(1 for r in comp_rooms if r in entrance_set)
        if n_exits >= min_exits:
            continue

        needed = min_exits - n_exits
        candidates = sorted(
            (r for r in comp_rooms if r not in entrance_set),
            key=lambda r: _distance_to_bbox(rooms[r].geometry, bbox),
        )
        for r in candidates[:needed]:
            g.add_edge(r, r, connectivity=Connectivity.ENTRANCE)
            entrance_set.add(r)
            report.repaired.append(f"added entrance at room {r}")
        if needed > len(candidates):
            report.unrepairable.append(
                f"component of size {len(comp_rooms)} needed {needed} exits "
                f"but only {len(candidates)} candidate rooms available"
            )

    return replace(plan, graph=g), report


def _distance_to_bbox(
    poly: Polygon, bbox: tuple[float, float, float, float]
) -> float:
    cx, cy = poly.centroid.x, poly.centroid.y
    return min(
        abs(cx - bbox[0]),
        abs(cx - bbox[2]),
        abs(cy - bbox[1]),
        abs(cy - bbox[3]),
    )


# ---------------------------------------------------------------------------
# dead_end_corridor — connect dangling corridors to nearest non-corridor room
# ---------------------------------------------------------------------------


def repair_dead_end_corridor(
    plan: PlanGraph, *, max_dead_end_length_m: float = 6.1
) -> tuple[PlanGraph, RepairReport]:
    """Add a synthetic passage from each over-long dead-end corridor to its
    nearest non-corridor neighbor.

    A "dead-end" is a corridor node with degree <= 1 whose major axis exceeds
    ``max_dead_end_length_m``. We add a passage edge to the geometrically
    nearest non-corridor room — this models punching a door through a wall
    rather than redrawing geometry, so it never invalidates polygons.
    """
    report = RepairReport()
    rooms = plan.rooms()
    g = plan.graph.copy()
    scale = _scale(plan)

    for nid in _corridor_nodes(plan):
        if g.degree(nid) > 1:
            continue
        room = rooms[nid]
        length_m = _polygon_max_dimension(room.geometry) * scale
        if length_m <= max_dead_end_length_m:
            continue

        target = _nearest_non_corridor(plan, nid)
        if target is None:
            report.unrepairable.append(
                f"corridor {nid} ({length_m:.2f} m) has no reachable non-corridor neighbor"
            )
            continue
        if g.has_edge(nid, target):
            # Already connected via something — only the connectivity type
            # was wrong (e.g. ENTRANCE self-loop). Skip.
            continue
        g.add_edge(nid, target, connectivity=Connectivity.PASSAGE)
        report.repaired.append(
            f"connected dead-end corridor {nid} to room {target} (passage)"
        )

    return replace(plan, graph=g), report


def _nearest_non_corridor(plan: PlanGraph, nid: Any) -> Any | None:
    rooms = plan.rooms()
    src = rooms[nid].geometry
    best, best_d = None, float("inf")
    for other, room in rooms.items():
        if other == nid or room.room_type == "Corridor":
            continue
        d = src.distance(room.geometry)
        if d < best_d:
            best, best_d = other, d
    return best


# ---------------------------------------------------------------------------
# wheelchair_turn_radius — flag offending rooms; full geometric repair is
# Stage-4 work. We report unrepairable so the sampler can resample.
# ---------------------------------------------------------------------------


def repair_wheelchair_turn_radius(
    plan: PlanGraph,
    *,
    min_diameter_m: float = 1.50,
    applies_to_room_types: tuple[str, ...] = ("Bathroom", "Livingroom", "Kitchen"),
) -> tuple[PlanGraph, RepairReport]:
    """Report rooms that fail the inscribed-circle test.

    Rectifying the turn-radius failure requires changing room *geometry* —
    that is out of scope for hard repair (it would invalidate any
    surrounding constraints). We surface it so callers can either resample
    or accept the violation.
    """
    report = RepairReport()
    scale = _scale(plan)
    for nid, room in plan.rooms().items():
        if room.room_type not in applies_to_room_types:
            continue
        diam = _polygon_inscribed_circle_diameter(room.geometry) * scale
        if diam + 1e-6 < min_diameter_m:
            report.unrepairable.append(
                f"{room.room_type} {nid}: inscribed diameter "
                f"{diam:.2f} m < required {min_diameter_m:.2f} m"
            )
    if not report.unrepairable:
        report.notes.append("all in-scope rooms already meet the turn-radius minimum")
    return plan, report


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def apply_hard_repairs(
    plan: PlanGraph, jurisdiction_code: str
) -> tuple[PlanGraph, dict[str, RepairReport]]:
    """Run the three hard-repair routines using a jurisdiction's parameters.

    Returns the (possibly modified) plan and a per-rule report dict.
    """
    spec = get_jurisdiction(jurisdiction_code)
    params = dict(spec.rules)

    reports: dict[str, RepairReport] = {}

    if "at_least_n_exits" in params:
        plan, reports["at_least_n_exits"] = repair_at_least_n_exits(
            plan, min_exits=params["at_least_n_exits"].get("min_exits", 1)
        )

    if "dead_end_corridor" in params:
        plan, reports["dead_end_corridor"] = repair_dead_end_corridor(
            plan,
            max_dead_end_length_m=params["dead_end_corridor"].get(
                "max_dead_end_length_m", 6.1
            ),
        )

    if "wheelchair_turn_radius" in params:
        plan, reports["wheelchair_turn_radius"] = repair_wheelchair_turn_radius(
            plan,
            min_diameter_m=params["wheelchair_turn_radius"].get(
                "min_diameter_m", 1.50
            ),
            applies_to_room_types=tuple(
                params["wheelchair_turn_radius"].get(
                    "applies_to_room_types",
                    ("Bathroom", "Livingroom", "Kitchen"),
                )
            ),
        )

    return plan, reports


__all__ = [
    "RepairReport",
    "apply_hard_repairs",
    "repair_at_least_n_exits",
    "repair_dead_end_corridor",
    "repair_wheelchair_turn_radius",
]
