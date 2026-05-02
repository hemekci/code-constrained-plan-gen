"""Building-code rule-checker module.

Public API: rules registered via `@register_rule(name)` are accessible by
name through `build_rule(name, **kwargs)` or via a JurisdictionSpec.

Importing this module triggers registration of the bundled rules.
"""

from __future__ import annotations

from .accessibility import CorridorMinWidth, DoorWidth
from .base import (
    Connectivity,
    PlanGraph,
    RoomNode,
    Rule,
    RuleClass,
    RuleResult,
    RULE_REGISTRY,
    build_rule,
    register_rule,
)
from .differentiable import (
    TensorRepr,
    compliance_energy_for_plan,
    corridor_width_energy,
    door_width_energy,
    egress_distance_energy,
    plan_to_tensor_repr,
    total_compliance_energy,
)
from .egress import DeadEndCorridor, EgressTravelDistance
from .jurisdictions import JURISDICTIONS, JurisdictionSpec, get_jurisdiction


def evaluate_jurisdiction(plan: PlanGraph, jurisdiction_code: str) -> dict:
    """Run every rule in a jurisdiction; return per-rule results plus aggregate.

    Returns a dict shaped:
        {
            "jurisdiction": "TR",
            "results": {rule_name: RuleResult, ...},
            "aggregate": {
                "all_passed": bool,
                "n_passed": int,
                "n_total": int,
                "compliance_score": float in [0, 1],  # weighted mean of soft scores
            },
        }
    """
    spec = get_jurisdiction(jurisdiction_code)
    rules = spec.build()
    per_rule = {r.name: r.check(plan) for r in rules}
    n_total = len(per_rule)
    n_passed = sum(1 for r in per_rule.values() if r.passed)
    compliance_score = (
        sum(r.score for r in per_rule.values()) / n_total if n_total else 0.0
    )
    return {
        "jurisdiction": jurisdiction_code,
        "results": per_rule,
        "aggregate": {
            "all_passed": n_passed == n_total,
            "n_passed": n_passed,
            "n_total": n_total,
            "compliance_score": compliance_score,
        },
    }


__all__ = [
    "Connectivity",
    "CorridorMinWidth",
    "DeadEndCorridor",
    "DoorWidth",
    "EgressTravelDistance",
    "JURISDICTIONS",
    "JurisdictionSpec",
    "PlanGraph",
    "RoomNode",
    "Rule",
    "RuleClass",
    "RuleResult",
    "RULE_REGISTRY",
    "TensorRepr",
    "build_rule",
    "compliance_energy_for_plan",
    "corridor_width_energy",
    "door_width_energy",
    "egress_distance_energy",
    "evaluate_jurisdiction",
    "get_jurisdiction",
    "plan_to_tensor_repr",
    "register_rule",
    "total_compliance_energy",
]
