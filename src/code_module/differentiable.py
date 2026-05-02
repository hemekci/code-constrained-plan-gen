"""Differentiable energy variants of the rule checks.

Stage 3 of the project plugs these into a Universal-Guidance / FreeDoM-style
sampling loop: at each diffusion step we predict ``x_hat_0`` (Tweedie estimate),
evaluate ``soft_compliance_energy(x_hat_0; jurisdiction)``, and add its gradient
to the score.

The differentiable energies operate on a tensor representation of a plan
rather than the shapely-based ``PlanGraph`` used elsewhere; converting between
the two is the responsibility of ``plan_to_tensor_repr`` (TBD in Stage 3 once
we lock the corner-coordinate convention used by HouseDiffusion).

For now, these functions are unit-tested with hand-crafted tensor inputs and
verified to:
- return zero (or near-zero) energy on compliant inputs,
- return strictly positive energy on non-compliant inputs,
- have non-zero gradients pointing in the direction that *reduces* the energy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import networkx as nx
import torch

if TYPE_CHECKING:  # avoid hard runtime dep on shapely for callers using only the tensor API
    from shapely.geometry import Polygon

    from .base import PlanGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PlanGraph -> tensor bridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorRepr:
    """Tensor representation of a PlanGraph used by the differentiable energies.

    Fields are aligned so they can be passed directly into
    ``total_compliance_energy``.
    """

    door_corners: torch.Tensor  # (N_doors, 4, 2)
    corridor_corners: torch.Tensor  # (M_corridors, 4, 2)
    pairwise_distances: torch.Tensor  # (R, R) shortest-path in meters
    entrance_mask: torch.Tensor  # (R,) bool
    room_node_ids: tuple  # ordering of rooms; same order as rows/cols of distances


def _polygon_rotated_rect_corners(polygon: "Polygon") -> Optional[list[tuple[float, float]]]:
    """Return the 4 corners of the rotated minimum-area bounding rectangle, or None."""
    if polygon.is_empty:
        return None
    min_box = polygon.minimum_rotated_rectangle
    coords = list(min_box.exterior.coords)
    if len(coords) < 5:  # closed ring expected
        return None
    return [(float(x), float(y)) for x, y in coords[:4]]


def plan_to_tensor_repr(
    plan: "PlanGraph",
    *,
    corridor_room_type: str = "Corridor",
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> TensorRepr:
    """Convert a ``PlanGraph`` into the tensors consumed by the energies.

    - ``door_corners`` are gathered from edge ``door_geometry`` polygons (where
      present); pickle-only graphs without door geometry produce an empty tensor.
    - ``corridor_corners`` come from rotated min-area boxes of every node whose
      ``room_type`` matches ``corridor_room_type``.
    - ``pairwise_distances`` is the all-pairs shortest path through the graph,
      with edges weighted by centroid distance (in meters via
      ``plan.unit_scale_m_per_unit``); disconnected pairs become +inf.
    - ``entrance_mask`` flags rooms touched by at least one ``ENTRANCE`` edge.
    """
    rooms = plan.rooms()
    room_ids = list(rooms.keys())
    n_rooms = len(room_ids)

    # Door corners
    door_corner_lists: list[list[tuple[float, float]]] = []
    for _, _, edata in plan.graph.edges(data=True):
        door_geom = edata.get("door_geometry")
        if door_geom is None:
            continue
        corners = _polygon_rotated_rect_corners(door_geom)
        if corners is None:
            continue
        door_corner_lists.append(corners)

    if door_corner_lists:
        door_corners = torch.tensor(door_corner_lists, dtype=dtype, device=device)
    else:
        door_corners = torch.empty((0, 4, 2), dtype=dtype, device=device)

    # Corridor corners
    corridor_corner_lists: list[list[tuple[float, float]]] = []
    for _, room in rooms.items():
        if room.room_type != corridor_room_type:
            continue
        corners = _polygon_rotated_rect_corners(room.geometry)
        if corners is None:
            continue
        corridor_corner_lists.append(corners)

    if corridor_corner_lists:
        corridor_corners = torch.tensor(corridor_corner_lists, dtype=dtype, device=device)
    else:
        corridor_corners = torch.empty((0, 4, 2), dtype=dtype, device=device)

    # Pairwise shortest-path distances
    weighted = nx.Graph()
    for nid in room_ids:
        weighted.add_node(nid)
    for u, v, edata in plan.graph.edges(data=True):
        if u not in rooms or v not in rooms:
            continue
        cu = rooms[u].centroid
        cv = rooms[v].centroid
        d_units = math.hypot(cu[0] - cv[0], cu[1] - cv[1])
        d_m = d_units * plan.unit_scale_m_per_unit
        weighted.add_edge(u, v, weight=d_m)

    distances = torch.full((n_rooms, n_rooms), float("inf"), dtype=dtype, device=device)
    for i, u in enumerate(room_ids):
        try:
            paths = nx.single_source_dijkstra_path_length(weighted, u, weight="weight")
        except nx.NodeNotFound:
            continue
        for v, d in paths.items():
            if v in rooms:
                j = room_ids.index(v)
                distances[i, j] = d

    # Entrance mask
    entrance_ids = set(plan.entrance_nodes())
    entrance_mask = torch.tensor(
        [nid in entrance_ids for nid in room_ids],
        dtype=torch.bool,
        device=device,
    )

    return TensorRepr(
        door_corners=door_corners,
        corridor_corners=corridor_corners,
        pairwise_distances=distances,
        entrance_mask=entrance_mask,
        room_node_ids=tuple(room_ids),
    )


# ---------------------------------------------------------------------------
# Door-width energy
# ---------------------------------------------------------------------------


def door_width_energy(
    door_corners: torch.Tensor,
    min_width_m: float,
) -> torch.Tensor:
    """Soft energy: ``sum(relu(min_width - door_long_side)^2)``.

    Parameters
    ----------
    door_corners
        Tensor of shape ``(N, 4, 2)`` — N doors, each as 4 corners ``(x, y)``
        in meters. Corner order is the rotated-rectangle order (consecutive
        corners are adjacent sides).
    min_width_m
        Required door width in meters.

    Returns
    -------
    Scalar tensor with the summed quadratic shortfall energy. Zero when every
    door's longer side >= ``min_width_m``.
    """
    if door_corners.numel() == 0:
        return torch.zeros((), dtype=door_corners.dtype, device=door_corners.device)
    # Side lengths between consecutive corners (cyclic): shape (N, 4)
    sides = torch.linalg.norm(
        door_corners - door_corners.roll(shifts=-1, dims=1),
        dim=-1,
    )
    # The clearance width is the longer of the two distinct side lengths;
    # for a rotated rectangle these are sides[:, 0] and sides[:, 1].
    long_side = torch.maximum(sides[:, 0], sides[:, 1])
    shortfall = torch.relu(min_width_m - long_side)
    return (shortfall * shortfall).sum()


# ---------------------------------------------------------------------------
# Corridor-width energy
# ---------------------------------------------------------------------------


def corridor_width_energy(
    corridor_corners: torch.Tensor,
    min_width_m: float,
) -> torch.Tensor:
    """Soft energy: ``sum(relu(min_width - corridor_short_side)^2)``.

    The corridor's narrowest passable width is the shorter side of its rotated
    minimum-area bounding rectangle; this is the dual of ``door_width_energy``.

    Parameters
    ----------
    corridor_corners
        Tensor of shape ``(M, 4, 2)`` — M corridor rooms as rotated rectangles.
    min_width_m
        Required corridor width in meters.
    """
    if corridor_corners.numel() == 0:
        return torch.zeros((), dtype=corridor_corners.dtype, device=corridor_corners.device)
    sides = torch.linalg.norm(
        corridor_corners - corridor_corners.roll(shifts=-1, dims=1),
        dim=-1,
    )
    short_side = torch.minimum(sides[:, 0], sides[:, 1])
    shortfall = torch.relu(min_width_m - short_side)
    return (shortfall * shortfall).sum()


# ---------------------------------------------------------------------------
# Egress travel-distance energy (soft-min over paths)
# ---------------------------------------------------------------------------


def soft_min(values: torch.Tensor, beta: float = 5.0, dim: int = -1) -> torch.Tensor:
    """Differentiable soft minimum via the log-sum-exp trick.

    ``soft_min(x) = -1/beta * logsumexp(-beta * x)``. Larger ``beta`` makes the
    approximation sharper; ``beta=5`` gives a useful gradient at typical
    egress-distance scales (tens of meters) without saturating.
    """
    return -torch.logsumexp(-beta * values, dim=dim) / beta


def egress_distance_energy(
    pairwise_distances: torch.Tensor,
    entrance_mask: torch.Tensor,
    max_distance_m: float,
    soft_min_beta: float = 5.0,
) -> torch.Tensor:
    """Soft energy: ``sum(relu(soft_min(d_to_entrance) - max_distance)^2)``.

    Parameters
    ----------
    pairwise_distances
        Tensor of shape ``(R, R)`` of room-to-room shortest-path distances in
        meters. Diagonal is zero. Disconnected pairs may be ``+inf`` or a
        large finite value; if ``+inf`` the soft-min still works via masking.
    entrance_mask
        Bool tensor of shape ``(R,)`` — True where the room has an entrance
        edge.
    max_distance_m
        Maximum allowed travel distance in meters.
    soft_min_beta
        Sharpness of the soft-min over candidate entrances.
    """
    R = pairwise_distances.shape[0]
    if R == 0 or not entrance_mask.any():
        return torch.zeros((), dtype=pairwise_distances.dtype, device=pairwise_distances.device)

    # For each room (rows), distances to all entrances (columns where mask is True).
    # We mask non-entrance columns by setting them to a large constant so they
    # don't dominate the soft-min.
    big = torch.full_like(pairwise_distances, fill_value=1e6)
    masked = torch.where(entrance_mask.unsqueeze(0), pairwise_distances, big)
    # Per-room soft-min distance to any entrance
    d_to_entrance = soft_min(masked, beta=soft_min_beta, dim=-1)
    # Skip the entrance rooms themselves (their distance is 0 → trivially compliant)
    overflow = torch.relu(d_to_entrance - max_distance_m)
    return (overflow * overflow).sum()


# ---------------------------------------------------------------------------
# Aggregate energy
# ---------------------------------------------------------------------------


def total_compliance_energy(
    *,
    door_corners: torch.Tensor,
    corridor_corners: torch.Tensor,
    pairwise_distances: torch.Tensor,
    entrance_mask: torch.Tensor,
    door_min_width_m: float,
    corridor_min_width_m: float,
    egress_max_distance_m: float,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Convenience aggregator: weighted sum of the three differentiable energies.

    Use this as the energy function in Universal-Guidance / FreeDoM-style
    sampling: ``grad = -lambda * autograd.grad(total_compliance_energy, x_hat)``.
    """
    w_door, w_corr, w_egr = weights
    e_door = door_width_energy(door_corners, door_min_width_m)
    e_corr = corridor_width_energy(corridor_corners, corridor_min_width_m)
    e_egr = egress_distance_energy(
        pairwise_distances, entrance_mask, egress_max_distance_m
    )
    return w_door * e_door + w_corr * e_corr + w_egr * e_egr


def compliance_energy_for_plan(
    plan: "PlanGraph",
    *,
    door_min_width_m: float,
    corridor_min_width_m: float,
    egress_max_distance_m: float,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """One-shot: convert a ``PlanGraph`` and return the soft compliance energy.

    Useful for sanity checks against the shapely-based hard rule check and as
    a reference implementation for Stage 3's sampling-time guidance loop.
    """
    repr = plan_to_tensor_repr(plan, dtype=dtype, device=device)
    distances = repr.pairwise_distances
    if torch.isinf(distances).any():
        # Replace +inf with a large finite penalty so soft-min stays well-defined
        big = torch.tensor(1e6, dtype=distances.dtype, device=distances.device)
        distances = torch.where(torch.isinf(distances), big, distances)
    return total_compliance_energy(
        door_corners=repr.door_corners,
        corridor_corners=repr.corridor_corners,
        pairwise_distances=distances,
        entrance_mask=repr.entrance_mask,
        door_min_width_m=door_min_width_m,
        corridor_min_width_m=corridor_min_width_m,
        egress_max_distance_m=egress_max_distance_m,
        weights=weights,
    )


__all__ = [
    "TensorRepr",
    "compliance_energy_for_plan",
    "corridor_width_energy",
    "door_width_energy",
    "egress_distance_energy",
    "plan_to_tensor_repr",
    "soft_min",
    "total_compliance_energy",
]
