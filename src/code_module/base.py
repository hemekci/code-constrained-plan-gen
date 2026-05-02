"""Base abstractions for building-code rule checking on floor plan graphs.

Plan representation aligns with the MSD (van Engelenburg et al., ECCV 2024)
graph-first convention: nodes are rooms with shapely polygon geometry,
edges encode connectivity (passage / door / entrance).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional

import networkx as nx
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


class RuleClass(IntEnum):
    """Solihin & Eastman 2015 rule classification."""

    SINGLE_ATTRIBUTE = 1  # e.g. "door width >= 0.9 m"
    SIMPLE_DERIVED = 2  # e.g. "corridor min width via polygon thinning"
    EXTENDED_STRUCTURE = 3  # e.g. "egress travel distance via graph"
    PROOF_REQUIRED = 4  # simulation / optimization / multi-step inference


class Connectivity(str, Enum):
    """MSD edge connectivity types (from data/MSD/graphs.py)."""

    PASSAGE = "passage"
    DOOR = "door"
    ENTRANCE = "entrance"


@dataclass(frozen=True)
class RoomNode:
    """One room as a node attribute on the plan graph."""

    geometry: Polygon
    room_type: str
    centroid: tuple[float, float] = field(default=(0.0, 0.0))


@dataclass(frozen=True)
class PlanGraph:
    """A floor plan as a connectivity graph plus geometry.

    Wraps a `networkx.Graph` so we have a typed surface and can attach
    plan-level metadata such as the building boundary and unit scale.
    """

    graph: nx.Graph
    boundary: Optional[Polygon] = None
    unit_scale_m_per_unit: float = 1.0  # multiply graph coords by this to get meters
    plan_id: Optional[str] = None

    def rooms(self) -> dict[Any, RoomNode]:
        """Return node-id -> RoomNode mapping."""
        result: dict[Any, RoomNode] = {}
        for nid, attrs in self.graph.nodes(data=True):
            geom = attrs.get("geometry")
            rtype = attrs.get("room_type")
            if geom is None or rtype is None:
                continue
            centroid_attr = attrs.get("centroid", (0.0, 0.0))
            centroid: tuple[float, float] = (
                tuple(centroid_attr) if hasattr(centroid_attr, "__iter__") else (0.0, 0.0)  # type: ignore[arg-type]
            )
            result[nid] = RoomNode(geometry=geom, room_type=rtype, centroid=centroid)
        return result

    def entrance_nodes(self) -> list[Any]:
        """Node ids that have at least one 'entrance' connectivity edge."""
        ids: list[Any] = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("connectivity") == Connectivity.ENTRANCE:
                ids.extend((u, v))
        return list(dict.fromkeys(ids))


@dataclass(frozen=True)
class RuleResult:
    """Outcome of evaluating one rule on one plan."""

    rule_name: str
    passed: bool
    score: float = 1.0
    """Soft compliance score in [0, 1]; 1.0 = fully compliant.

    For hard rules without a notion of partial credit this equals `1.0` if
    `passed` else `0.0`.
    """
    details: dict[str, Any] = field(default_factory=dict)
    """Per-rule diagnostics: which element violated, by how much, etc."""


class Rule(ABC):
    """Abstract base class for one building-code rule."""

    name: str = "unnamed-rule"
    rule_class: RuleClass = RuleClass.SINGLE_ATTRIBUTE

    @abstractmethod
    def check(self, plan: PlanGraph) -> RuleResult:
        """Evaluate the rule on a plan; return a RuleResult."""

    def energy(self, plan: PlanGraph) -> float:
        """Non-negative soft penalty; 0 means fully compliant.

        Default implementation returns `0.0` if the hard check passes else
        `1.0`. Subclasses should override with a smooth relaxation when a
        differentiable energy is required for guidance during sampling.
        """
        result = self.check(plan)
        return 0.0 if result.passed else 1.0


# ---------------------------------------------------------------------------
# Registry (Factory pattern)
# ---------------------------------------------------------------------------

RULE_REGISTRY: dict[str, type[Rule]] = {}


def register_rule(name: str):
    """Decorator that registers a Rule subclass under `name`."""

    def decorator(cls: type[Rule]) -> type[Rule]:
        if name in RULE_REGISTRY:
            logger.warning("Rule %s is already registered; overwriting.", name)
        RULE_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def build_rule(name: str, **kwargs: Any) -> Rule:
    """Instantiate a Rule by name with optional keyword arguments."""
    if name not in RULE_REGISTRY:
        raise KeyError(
            f"Rule '{name}' not registered. Available: {sorted(RULE_REGISTRY)}"
        )
    return RULE_REGISTRY[name](**kwargs)


__all__ = [
    "Connectivity",
    "PlanGraph",
    "RoomNode",
    "Rule",
    "RuleClass",
    "RuleResult",
    "RULE_REGISTRY",
    "build_rule",
    "register_rule",
]
