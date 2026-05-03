"""Fire egress rules — graph-derived (Solihin-Eastman Class 3).

Implements representative egress rules:
- EgressTravelDistance: shortest weighted path from each room to any entrance <= max
- DeadEndCorridor: corridor nodes with degree <= 1 are flagged as dead-ends

Edge weights are taken from polygon centroid distances; this is a coarse
approximation that we may refine later (e.g., medial-axis path through corridor).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import networkx as nx

from .base import PlanGraph, Rule, RuleClass, RuleResult, register_rule

logger = logging.getLogger(__name__)


def _euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _build_weighted_graph(plan: PlanGraph) -> nx.Graph:
    """Return a copy of the plan graph with edges weighted by centroid distance.

    Distances are converted to meters via `plan.unit_scale_m_per_unit`.
    Disconnected nodes remain disconnected (no synthetic edges added).
    """
    rooms = plan.rooms()
    g = nx.Graph()
    for nid in plan.graph.nodes():
        g.add_node(nid)
    for u, v, data in plan.graph.edges(data=True):
        if u not in rooms or v not in rooms:
            continue
        d_units = _euclidean(rooms[u].centroid, rooms[v].centroid)
        d_m = d_units * plan.unit_scale_m_per_unit
        g.add_edge(u, v, weight=d_m, **data)
    return g


@register_rule("egress_travel_distance")
@dataclass
class EgressTravelDistance(Rule):
    """Shortest path from any room to any entrance must be <= `max_distance_m`.

    Defaults to 30 m (a typical residential single-route limit; non-sprinklered
    occupancies in some codes go down to 23 m and sprinklered go up to 60 m).
    Tighten/loosen via jurisdiction.
    """

    max_distance_m: float = 30.0
    rule_class: RuleClass = RuleClass.EXTENDED_STRUCTURE

    def check(self, plan: PlanGraph) -> RuleResult:
        rooms = plan.rooms()
        entrances = plan.entrance_nodes()
        if not entrances:
            return RuleResult(
                rule_name=self.name,
                passed=False,
                score=0.0,
                details={"reason": "no entrance edge present"},
            )
        weighted = _build_weighted_graph(plan)
        violations: list[dict[str, float]] = []
        per_room: dict[str, float] = {}
        for nid in rooms.keys():
            if nid in entrances:
                per_room[str(nid)] = 0.0
                continue
            best = math.inf
            for ent in entrances:
                try:
                    d = nx.shortest_path_length(
                        weighted, source=nid, target=ent, weight="weight"
                    )
                    best = min(best, d)
                except nx.NetworkXNoPath:
                    continue
                except nx.NodeNotFound:
                    continue
            per_room[str(nid)] = best if math.isfinite(best) else math.inf
            if not math.isfinite(best) or best > self.max_distance_m:
                violations.append(
                    {
                        "node": nid,
                        "distance_m": best if math.isfinite(best) else -1.0,
                        "max_distance_m": self.max_distance_m,
                    }
                )
        passed = len(violations) == 0
        score = (
            1.0
            if passed
            else max(0.0, 1.0 - len(violations) / max(len(rooms), 1))
        )
        return RuleResult(
            rule_name=self.name,
            passed=passed,
            score=score,
            details={
                "violations": violations,
                "per_room_distance_m": per_room,
                "n_entrances": len(entrances),
            },
        )


@register_rule("dead_end_corridor")
@dataclass
class DeadEndCorridor(Rule):
    """No corridor node may form a dead-end (degree <= 1) above
    `max_dead_end_length_m`.

    Default length 6.1 m (≈ 20 ft, IBC residential typical).
    """

    max_dead_end_length_m: float = 6.1
    corridor_room_type: str = "Corridor"
    rule_class: RuleClass = RuleClass.EXTENDED_STRUCTURE

    def check(self, plan: PlanGraph) -> RuleResult:
        rooms = plan.rooms()
        violations: list[dict[str, float]] = []
        n_corridors = 0
        for nid, room in rooms.items():
            if room.room_type != self.corridor_room_type:
                continue
            n_corridors += 1
            degree = plan.graph.degree(nid)
            if degree > 1:
                continue
            # estimate the dead-end "length" as polygon major axis
            min_box = room.geometry.minimum_rotated_rectangle
            coords = list(min_box.exterior.coords)
            sides_units: list[float] = []
            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:], strict=False):
                sides_units.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            length_units = max(sides_units) if sides_units else 0.0
            length_m = length_units * plan.unit_scale_m_per_unit
            if length_m > self.max_dead_end_length_m:
                violations.append(
                    {
                        "node": nid,
                        "length_m": length_m,
                        "max_length_m": self.max_dead_end_length_m,
                    }
                )
        passed = len(violations) == 0
        score = (
            1.0
            if passed
            else max(0.0, 1.0 - len(violations) / max(n_corridors, 1))
        )
        return RuleResult(
            rule_name=self.name,
            passed=passed,
            score=score,
            details={
                "violations": violations,
                "n_corridors": n_corridors,
            },
        )


@register_rule("at_least_n_exits")
@dataclass
class AtLeastNExits(Rule):
    """Each connected component (apartment unit) must have ``min_exits`` or
    more entrance edges (Class 3-4: proof-of-existence over the graph).

    For most residential codes, ``min_exits=1`` is sufficient per apartment.
    Larger occupancies (multi-storey, large floor area) often require 2; we
    expose ``min_exits`` so each jurisdiction can override.
    """

    min_exits: int = 1
    rule_class: RuleClass = RuleClass.PROOF_REQUIRED

    def check(self, plan: PlanGraph) -> RuleResult:
        rooms = plan.rooms()
        if not rooms:
            return RuleResult(
                rule_name=self.name,
                passed=False,
                score=0.0,
                details={"reason": "no rooms"},
            )

        # Walk connected components on the room subgraph (door + passage + entrance edges)
        room_only = nx.Graph()
        room_only.add_nodes_from(rooms.keys())
        for u, v, _ in plan.graph.edges(data=True):
            if u in rooms and v in rooms and u != v:
                room_only.add_edge(u, v)

        entrance_set = set(plan.entrance_nodes())
        violations: list[dict] = []
        components: list[dict] = []
        for comp in nx.connected_components(room_only):
            comp_rooms = [r for r in comp if r in rooms]
            n_exits = sum(1 for r in comp_rooms if r in entrance_set)
            comp_info = {
                "size": len(comp_rooms),
                "n_exits": n_exits,
                "rooms": comp_rooms[:5],
            }
            components.append(comp_info)
            if n_exits < self.min_exits:
                violations.append(comp_info)

        passed = len(violations) == 0
        score = (
            1.0
            if passed
            else max(0.0, 1.0 - len(violations) / max(len(components), 1))
        )
        return RuleResult(
            rule_name=self.name,
            passed=passed,
            score=score,
            details={
                "violations": violations,
                "components": components,
                "n_components": len(components),
                "min_exits_required": self.min_exits,
            },
        )


__all__ = ["AtLeastNExits", "DeadEndCorridor", "EgressTravelDistance"]
