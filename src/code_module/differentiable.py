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

import torch

logger = logging.getLogger(__name__)


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


__all__ = [
    "corridor_width_energy",
    "door_width_energy",
    "egress_distance_energy",
    "soft_min",
    "total_compliance_energy",
]
