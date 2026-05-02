"""Jurisdiction-specific rule sets and parameter overrides.

Each jurisdiction is an immutable mapping from rule name to keyword arguments
used to instantiate that rule. Adding a new jurisdiction is a one-line addition
to JURISDICTIONS.
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


# Reasonable starting configurations; numbers can be tightened per the actual
# regulation text in Stage 2 once we cross-reference TS 9111, ADA, ISO 21542.
JURISDICTIONS: dict[str, JurisdictionSpec] = {
    "TR": JurisdictionSpec(
        name="Türkiye (TS 9111)",
        rules=(
            ("door_width", {"min_width_m": 0.90}),
            ("corridor_min_width", {"min_width_m": 1.20}),
            ("egress_travel_distance", {"max_distance_m": 25.0}),
            ("dead_end_corridor", {"max_dead_end_length_m": 6.0}),
        ),
    ),
    "US": JurisdictionSpec(
        name="United States (ADA + IBC residential)",
        rules=(
            ("door_width", {"min_width_m": 0.815}),  # ADA 32"
            ("corridor_min_width", {"min_width_m": 0.915}),  # IBC residential 36"
            ("egress_travel_distance", {"max_distance_m": 38.1}),  # 125 ft sprinklered
            ("dead_end_corridor", {"max_dead_end_length_m": 6.1}),  # 20 ft
        ),
    ),
    "ISO": JurisdictionSpec(
        name="ISO 21542 (international accessibility)",
        rules=(
            ("door_width", {"min_width_m": 0.85}),
            ("corridor_min_width", {"min_width_m": 1.20}),
            ("egress_travel_distance", {"max_distance_m": 30.0}),
            ("dead_end_corridor", {"max_dead_end_length_m": 7.5}),
        ),
    ),
}


def get_jurisdiction(code: str) -> JurisdictionSpec:
    if code not in JURISDICTIONS:
        raise KeyError(
            f"Jurisdiction '{code}' not registered. Available: {sorted(JURISDICTIONS)}"
        )
    return JURISDICTIONS[code]


__all__ = ["JURISDICTIONS", "JurisdictionSpec", "get_jurisdiction"]
