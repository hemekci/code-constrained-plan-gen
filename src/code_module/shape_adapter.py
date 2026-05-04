"""Shape adapters that bridge backbone output tensors to rule-energy inputs.

The differentiable rule energies in ``code_module.differentiable`` consume
rotated-rectangle corner tensors of shape ``(..., 4, 2)``. Real diffusion
backbones produce per-polygon corner tensors with more vertices — for
example HouseDiffusion samples ``(B, N_rooms, 32, 2)``. These adapters
collapse the per-polygon dimension down to 4 rotated-min-bbox corners
in a way that preserves gradients.

Approach: PCA-based oriented bounding box. For each polygon we
    1. center on its mean,
    2. take SVD of the centered points to get principal axes,
    3. project the points onto each axis to get extents,
    4. emit the 4 corners in rotated-rect (consecutive-side) order.

This is fully differentiable (``torch.linalg.svd`` supports autograd) and
matches the true minimum-area bbox closely on convex-ish floor-plan
polygons, which is the regime HouseDiffusion produces.
"""

from __future__ import annotations

from typing import Tuple

import torch


def polygons_to_rotated_rect(polygons: torch.Tensor) -> torch.Tensor:
    """Reduce ``(..., V, 2)`` polygon vertices to ``(..., 4, 2)`` bbox corners.

    Parameters
    ----------
    polygons
        Tensor with shape ``(..., V, 2)`` where ``V >= 2``. Trailing two
        dims are the polygon's vertex list and the (x, y) coordinate.

    Returns
    -------
    Tensor of shape ``(..., 4, 2)`` — for each polygon, the four corners
    of its PCA-derived oriented bounding box, in rotated-rect order
    (so consecutive corners are adjacent sides, matching what
    ``door_width_energy`` and ``corridor_width_energy`` expect).
    """
    if polygons.shape[-1] != 2:
        raise ValueError(
            f"polygons last dim must be 2, got shape {tuple(polygons.shape)}"
        )
    if polygons.shape[-2] < 2:
        raise ValueError(
            f"polygons need at least 2 vertices, got {polygons.shape[-2]}"
        )

    centered = polygons - polygons.mean(dim=-2, keepdim=True)  # (..., V, 2)

    # SVD on the (V, 2) matrix per polygon. Vh has shape (..., 2, 2) — its
    # rows are the right-singular vectors (the principal axes).
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    axes = vh  # (..., 2, 2). axes[..., 0, :] = primary, axes[..., 1, :] = secondary

    # Project onto each axis: (..., V, 2) @ (..., 2, 2)^T -> (..., V, 2)
    projected = torch.matmul(centered, axes.transpose(-1, -2))
    p_min = projected.amin(dim=-2)  # (..., 2)
    p_max = projected.amax(dim=-2)  # (..., 2)

    # 4 rotated-rect corners in axis-local coords, consecutive-side order:
    #   (min_u, min_v), (max_u, min_v), (max_u, max_v), (min_u, max_v)
    u_min = p_min[..., 0:1]
    u_max = p_max[..., 0:1]
    v_min = p_min[..., 1:2]
    v_max = p_max[..., 1:2]
    local = torch.stack(
        [
            torch.cat([u_min, v_min], dim=-1),
            torch.cat([u_max, v_min], dim=-1),
            torch.cat([u_max, v_max], dim=-1),
            torch.cat([u_min, v_max], dim=-1),
        ],
        dim=-2,
    )  # (..., 4, 2)

    # Rotate back into world coords: corners_world = corners_local @ axes
    # plus the original mean.
    rotated = torch.matmul(local, axes)
    return rotated + polygons.mean(dim=-2, keepdim=True)


def collapse_rooms_to_doors(
    rooms: torch.Tensor, *, take: int | None = None
) -> torch.Tensor:
    """Reshape ``(B, N_rooms, 4, 2)`` -> ``(B*N_rooms, 4, 2)`` (or first ``take``).

    Convenience helper: the rule energies treat each ``(4, 2)`` block as a
    polygon to score, regardless of which room it came from. Passing
    ``take`` truncates to the first ``take`` polygons (e.g. to score
    only the smallest-N for door-shaped doors vs corridor-shaped corridors).
    """
    flat = rooms.reshape(-1, 4, 2)
    if take is not None:
        flat = flat[:take]
    return flat


def housediffusion_x_to_rect_corners(x: torch.Tensor) -> torch.Tensor:
    """Convenience: HouseDiffusion ``(B, N_rooms, 32, 2)`` -> ``(B*N_rooms, 4, 2)``.

    Drops the batch/room distinction so the rule-energy fn sees a flat
    stack of polygon rotated rectangles. Used by ``jurisdiction_energy_fn``
    when the backbone is HouseDiffusion.
    """
    if x.dim() < 3:
        raise ValueError(
            f"housediffusion x must be at least 3-D (B, V, 2) or "
            f"(B, N_rooms, V, 2); got shape {tuple(x.shape)}"
        )
    rect = polygons_to_rotated_rect(x)  # (..., 4, 2)
    return rect.reshape(-1, 4, 2)


__all__ = [
    "collapse_rooms_to_doors",
    "housediffusion_x_to_rect_corners",
    "polygons_to_rotated_rect",
]
