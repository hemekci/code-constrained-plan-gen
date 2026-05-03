"""Accessibility rules — TS 9111 / ADA / ISO 21542 inspired.

Implements representative Class 1-2 rules from Solihin & Eastman 2015:
- DoorWidth: each door polygon's minimum dimension >= threshold
- CorridorMinWidth: each corridor room's narrowest passable width >= threshold

Thresholds default to the most permissive of the three jurisdictions; subclasses
or jurisdiction-specific instantiations tighten the value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from shapely.geometry import Polygon

from .base import Connectivity, PlanGraph, Rule, RuleClass, RuleResult, register_rule

logger = logging.getLogger(__name__)


def _rotated_box_sides(poly: Polygon) -> tuple[float, float]:
    """Return (shorter_side, longer_side) of the polygon's rotated min-area box.

    Returns (0.0, 0.0) if the polygon is empty or degenerate. Used by both
    corridor-width (shorter side = narrowest passable width) and door-width
    (longer side = clearance opening) rules.
    """
    if poly.is_empty:
        return 0.0, 0.0
    min_box = poly.minimum_rotated_rectangle
    coords = list(min_box.exterior.coords)
    if len(coords) < 4:
        return 0.0, 0.0
    sides: list[float] = []
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:], strict=False):
        sides.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if not sides:
        return 0.0, 0.0
    return float(min(sides)), float(max(sides))


def _polygon_min_dimension(poly: Polygon) -> float:
    """Shorter side of the polygon's rotated min-area bounding rectangle."""
    return _rotated_box_sides(poly)[0]


def _polygon_max_dimension(poly: Polygon) -> float:
    """Longer side of the polygon's rotated min-area bounding rectangle."""
    return _rotated_box_sides(poly)[1]


@register_rule("door_width")
@dataclass
class DoorWidth(Rule):
    """Minimum clear opening width of each door (Class 1).

    `min_width_m`: required width in meters. Defaults to 0.80 m
    (ISO 21542 doorway clear width minimum). ADA requires 32 inches ≈ 0.81 m;
    TS 9111 requires 0.90 m for accessible main doors.
    """

    min_width_m: float = 0.80
    rule_class: RuleClass = RuleClass.SINGLE_ATTRIBUTE

    def check(self, plan: PlanGraph) -> RuleResult:
        violations: list[dict[str, float]] = []
        n_doors = 0
        for u, v, data in plan.graph.edges(data=True):
            if data.get("connectivity") not in (
                Connectivity.DOOR,
                Connectivity.ENTRANCE,
            ):
                continue
            n_doors += 1
            door_geom = data.get("door_geometry")
            if door_geom is None:
                # Door edge without explicit geometry; cannot check, skip with note.
                continue
            width_units = _polygon_max_dimension(door_geom)
            width_m = width_units * plan.unit_scale_m_per_unit
            if width_m + 1e-6 < self.min_width_m:
                violations.append(
                    {
                        "edge": (u, v),
                        "width_m": width_m,
                        "required_m": self.min_width_m,
                    }
                )
        passed = len(violations) == 0
        score = 1.0 if passed else max(0.0, 1.0 - len(violations) / max(n_doors, 1))
        return RuleResult(
            rule_name=self.name,
            passed=passed,
            score=score,
            details={"violations": violations, "n_doors": n_doors},
        )

    def energy(self, plan: PlanGraph) -> float:
        total = 0.0
        for _, _, data in plan.graph.edges(data=True):
            if data.get("connectivity") not in (
                Connectivity.DOOR,
                Connectivity.ENTRANCE,
            ):
                continue
            door_geom = data.get("door_geometry")
            if door_geom is None:
                continue
            width_m = _polygon_max_dimension(door_geom) * plan.unit_scale_m_per_unit
            shortfall = self.min_width_m - width_m
            if shortfall > 0:
                total += shortfall * shortfall
        return float(total)


@register_rule("corridor_min_width")
@dataclass
class CorridorMinWidth(Rule):
    """Each corridor's narrowest passable width >= threshold (Class 2).

    `min_width_m`: required corridor width in meters. ISO 21542 / TS 9111
    accessible corridor: 1.20 m. Tight residential corridors in older codes
    accept 0.90 m; we default to 1.20 m for the strict accessibility case.
    """

    min_width_m: float = 1.20
    corridor_room_type: str = "Corridor"
    rule_class: RuleClass = RuleClass.SIMPLE_DERIVED

    def check(self, plan: PlanGraph) -> RuleResult:
        violations: list[dict[str, float]] = []
        n_corridors = 0
        for nid, room in plan.rooms().items():
            if room.room_type != self.corridor_room_type:
                continue
            n_corridors += 1
            width_units = _polygon_min_dimension(room.geometry)
            width_m = width_units * plan.unit_scale_m_per_unit
            if width_m + 1e-6 < self.min_width_m:
                violations.append(
                    {
                        "node": nid,
                        "width_m": width_m,
                        "required_m": self.min_width_m,
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
                "skipped_no_corridor": n_corridors == 0,
            },
        )


def _polygon_inscribed_circle_diameter(poly: Polygon) -> float:
    """Diameter of the largest circle inscribed in the polygon.

    Uses shapely's maximum_inscribed_circle when available (>= 2.0). The
    inscribed-circle diameter is the standard differentiable proxy for
    "wheelchair turn space available" within a room.
    """
    if poly.is_empty:
        return 0.0
    try:
        # Tolerance trades precision for speed; 0.05 m is plenty for room-scale
        line = poly.maximum_inscribed_circle(tolerance=0.05)  # type: ignore[attr-defined]
        # The returned LineString has length = radius
        radius = line.length
        return float(2.0 * radius)
    except (AttributeError, Exception):  # noqa: BLE001
        # Fallback: distance from centroid to boundary, doubled (lower bound)
        try:
            return float(2.0 * poly.exterior.distance(poly.centroid))
        except Exception:  # noqa: BLE001
            return 0.0


@register_rule("wheelchair_turn_radius")
@dataclass
class WheelchairTurnRadius(Rule):
    """Each accessible room must contain a turning space whose inscribed
    circle diameter >= ``min_diameter_m`` (Class 2).

    Default 1.50 m is the most common accessibility-code minimum
    (ISO 21542, EN 17210, ADA, AS 1428.1). DIN 18040-2 R variant requires
    1.50 m as well. ``applies_to_room_types`` lets jurisdictions tighten or
    relax which rooms must comply (e.g. only Bathroom, only Bedroom +
    Livingroom, etc.).
    """

    min_diameter_m: float = 1.50
    applies_to_room_types: tuple[str, ...] = ("Bathroom", "Livingroom", "Kitchen")
    rule_class: RuleClass = RuleClass.SIMPLE_DERIVED

    def check(self, plan: PlanGraph) -> RuleResult:
        violations: list[dict] = []
        n_checked = 0
        for nid, room in plan.rooms().items():
            if room.room_type not in self.applies_to_room_types:
                continue
            n_checked += 1
            diameter_units = _polygon_inscribed_circle_diameter(room.geometry)
            diameter_m = diameter_units * plan.unit_scale_m_per_unit
            if diameter_m + 1e-6 < self.min_diameter_m:
                violations.append(
                    {
                        "node": nid,
                        "room_type": room.room_type,
                        "diameter_m": diameter_m,
                        "required_m": self.min_diameter_m,
                    }
                )
        passed = len(violations) == 0
        score = (
            1.0
            if passed
            else max(0.0, 1.0 - len(violations) / max(n_checked, 1))
        )
        return RuleResult(
            rule_name=self.name,
            passed=passed,
            score=score,
            details={
                "violations": violations,
                "n_checked": n_checked,
                "applies_to": list(self.applies_to_room_types),
            },
        )


__all__ = ["CorridorMinWidth", "DoorWidth", "WheelchairTurnRadius"]
