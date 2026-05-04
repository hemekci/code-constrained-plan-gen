"""Jurisdiction-specific rule sets and parameter overrides.

Each jurisdiction is an immutable mapping from rule name to keyword arguments
used to instantiate that rule. Adding a new jurisdiction is a one-line addition
to ``JURISDICTIONS``.

The thresholds below are **representative**, not authoritative. Real building
codes are far more nuanced — they branch on occupancy class, building height,
sprinklered status, accessible-route designation, etc. We use values that are
defensible for accessible residential apartments in each regime; the paper
explicitly notes this scoping choice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .base import Rule, build_rule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JurisdictionSpec:
    """One jurisdiction's rule selection and parameter overrides."""

    name: str
    rules: tuple[tuple[str, dict], ...]

    def build(self) -> list[Rule]:
        return [build_rule(rname, **rkwargs) for rname, rkwargs in self.rules]


# Representative jurisdictions covering a wide spectrum of strictness across
# accessible-residential building codes. Sourced from each code's accessibility
# main text plus its associated egress provisions; see the disclaimer above.
JURISDICTIONS: dict[str, JurisdictionSpec] = {
    "ISO": JurisdictionSpec(
        name="ISO 21542 (international accessibility)",
        rules=(
            ("door_width", {"min_width_m": 0.85}),
            ("corridor_min_width", {"min_width_m": 1.20}),
            ("egress_travel_distance", {"max_distance_m": 30.0}),
            ("dead_end_corridor", {"max_dead_end_length_m": 7.5}),
            ("at_least_n_exits", {"min_exits": 1}),
            ("wheelchair_turn_radius", {"min_diameter_m": 1.50}),
        ),
    ),
    "EU": JurisdictionSpec(
        name="European Union (EN 17210 + Eurocode-based egress)",
        rules=(
            ("door_width", {"min_width_m": 0.85}),
            ("corridor_min_width", {"min_width_m": 1.20}),
            ("egress_travel_distance", {"max_distance_m": 30.0}),
            ("dead_end_corridor", {"max_dead_end_length_m": 7.5}),
            ("at_least_n_exits", {"min_exits": 1}),
            ("wheelchair_turn_radius", {"min_diameter_m": 1.50}),
        ),
    ),
    "US": JurisdictionSpec(
        name="United States (ADA + IBC residential)",
        rules=(
            ("door_width", {"min_width_m": 0.815}),  # ADA 32"
            ("corridor_min_width", {"min_width_m": 0.915}),  # IBC residential 36"
            ("egress_travel_distance", {"max_distance_m": 38.1}),  # 125 ft sprinklered
            ("dead_end_corridor", {"max_dead_end_length_m": 6.1}),  # 20 ft
            ("at_least_n_exits", {"min_exits": 1}),  # IBC R-2 dwelling unit
            ("wheelchair_turn_radius", {"min_diameter_m": 1.525}),  # ADA 60"
        ),
    ),
    "UK": JurisdictionSpec(
        name="United Kingdom (Approved Document M + Approved Document B)",
        rules=(
            ("door_width", {"min_width_m": 0.80}),  # AD M dwellings
            ("corridor_min_width", {"min_width_m": 1.20}),  # AD M cat 2
            ("egress_travel_distance", {"max_distance_m": 30.0}),  # AD B residential
            ("dead_end_corridor", {"max_dead_end_length_m": 7.5}),
            ("at_least_n_exits", {"min_exits": 1}),
            ("wheelchair_turn_radius", {"min_diameter_m": 1.50}),  # AD M cat 2
        ),
    ),
    "DE": JurisdictionSpec(
        name="Germany (DIN 18040-2 + Bauordnung egress)",
        rules=(
            ("door_width", {"min_width_m": 0.90}),  # DIN 18040-2 wheelchair-accessible
            ("corridor_min_width", {"min_width_m": 1.20}),
            ("egress_travel_distance", {"max_distance_m": 35.0}),
            ("dead_end_corridor", {"max_dead_end_length_m": 7.5}),
            ("at_least_n_exits", {"min_exits": 1}),
            ("wheelchair_turn_radius", {"min_diameter_m": 1.50}),  # DIN 18040-2 R variant
        ),
    ),
    "TR": JurisdictionSpec(
        name="Türkiye (TS 9111)",
        rules=(
            ("door_width", {"min_width_m": 0.90}),
            ("corridor_min_width", {"min_width_m": 1.20}),
            ("egress_travel_distance", {"max_distance_m": 25.0}),
            ("dead_end_corridor", {"max_dead_end_length_m": 6.0}),
            ("at_least_n_exits", {"min_exits": 1}),
            ("wheelchair_turn_radius", {"min_diameter_m": 1.50}),
        ),
    ),
    "JP": JurisdictionSpec(
        name="Japan (Barrier-Free Law / 高齢者・障害者等の移動等の円滑化)",
        rules=(
            ("door_width", {"min_width_m": 0.80}),
            ("corridor_min_width", {"min_width_m": 1.20}),
            ("egress_travel_distance", {"max_distance_m": 30.0}),
            ("dead_end_corridor", {"max_dead_end_length_m": 7.5}),
            ("at_least_n_exits", {"min_exits": 1}),
            ("wheelchair_turn_radius", {"min_diameter_m": 1.40}),  # Heart Building Law
        ),
    ),
    "AU": JurisdictionSpec(
        name="Australia (AS 1428.1 + NCC residential)",
        rules=(
            ("door_width", {"min_width_m": 0.85}),  # AS 1428.1
            ("corridor_min_width", {"min_width_m": 1.00}),  # AS 1428.1 typical
            ("egress_travel_distance", {"max_distance_m": 40.0}),  # NCC sprinklered
            ("dead_end_corridor", {"max_dead_end_length_m": 6.0}),
            ("at_least_n_exits", {"min_exits": 1}),
            ("wheelchair_turn_radius", {"min_diameter_m": 1.55}),  # AS 1428.1
        ),
    ),
    "SG": JurisdictionSpec(
        name="Singapore (BCA Code on Accessibility 2019)",
        rules=(
            ("door_width", {"min_width_m": 0.85}),
            ("corridor_min_width", {"min_width_m": 1.20}),
            ("egress_travel_distance", {"max_distance_m": 30.0}),
            ("dead_end_corridor", {"max_dead_end_length_m": 7.5}),
            ("at_least_n_exits", {"min_exits": 1}),
            ("wheelchair_turn_radius", {"min_diameter_m": 1.50}),
        ),
    ),
}


def get_jurisdiction(code: str) -> JurisdictionSpec:
    if code not in JURISDICTIONS:
        raise KeyError(
            f"Jurisdiction '{code}' not registered. Available: {sorted(JURISDICTIONS)}"
        )
    return JURISDICTIONS[code]


def permissive_door_jurisdiction(code: str) -> JurisdictionSpec:
    """Return a jurisdiction with door_width restricted to accessible-route doors.

    This matches how real codes scope the door-width threshold (entrance,
    corridor, and primary-space doors only — not interior bedroom doors).
    Use this variant alongside the strict version to expose how much of the
    headline non-compliance number is explained by interior doors that are
    legally out-of-scope.
    """
    base = get_jurisdiction(code)
    rules = tuple(
        (rname, {**rkwargs, "scope": "accessible_route"})
        if rname == "door_width"
        else (rname, rkwargs)
        for rname, rkwargs in base.rules
    )
    return JurisdictionSpec(
        name=f"{base.name} — accessible-route door scope", rules=rules
    )


__all__ = [
    "JURISDICTIONS",
    "JurisdictionSpec",
    "get_jurisdiction",
    "permissive_door_jurisdiction",
]
