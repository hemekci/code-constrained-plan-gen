"""Tests for the polygon -> rotated-rect shape adapter."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_module import (  # noqa: E402
    door_width_energy,
    housediffusion_x_to_rect_corners,
    polygons_to_rotated_rect,
)


def test_axis_aligned_square_recovers_itself() -> None:
    """A 1 m axis-aligned square should round-trip through the adapter
    with the same rotated-rect bbox (up to corner ordering)."""
    pts = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    rect = polygons_to_rotated_rect(pts)
    assert rect.shape == (4, 2)
    # Bbox extents must be 1 m on each side.
    sides = (rect - rect.roll(-1, dims=0)).norm(dim=-1)
    assert torch.allclose(sides, torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64))


def test_rotated_rect_preserves_extents() -> None:
    """A rectangle rotated 30 degrees should still expose its 2.0 m x 0.7 m extents."""
    base = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 0.7], [0.0, 0.7]], dtype=torch.float64
    )
    theta = math.radians(30)
    R = torch.tensor(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]],
        dtype=torch.float64,
    )
    rotated_pts = base @ R.T
    rect = polygons_to_rotated_rect(rotated_pts)
    sides = (rect - rect.roll(-1, dims=0)).norm(dim=-1)
    long_side = float(sides.max())
    short_side = float(sides.min())
    assert abs(long_side - 2.0) < 1e-6
    assert abs(short_side - 0.7) < 1e-6


def test_polygon_with_32_vertices_collapses_to_4_corners() -> None:
    """HouseDiffusion's 32-vertex per-room layout must collapse to (..., 4, 2)."""
    # Sample 32 points around the perimeter of a 1.5 x 1.5 square.
    t = torch.linspace(0.0, 4.0, 33, dtype=torch.float64)[:-1]
    pts = []
    for ti in t:
        seg = int(ti.item()) % 4
        f = float(ti.item()) - seg
        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        x0, y0 = corners[seg]
        x1, y1 = corners[(seg + 1) % 4]
        pts.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
    polygon = torch.tensor(pts, dtype=torch.float64) * 1.5  # (32, 2)
    poly_batch = polygon.unsqueeze(0).unsqueeze(0)  # (B=1, N_rooms=1, 32, 2)

    rect = polygons_to_rotated_rect(poly_batch)
    assert rect.shape == (1, 1, 4, 2)

    flat = housediffusion_x_to_rect_corners(poly_batch)
    assert flat.shape == (1, 4, 2)


def test_adapter_keeps_gradients_flowing() -> None:
    pts = torch.randn(1, 1, 32, 2, dtype=torch.float64, requires_grad=True)
    rect = housediffusion_x_to_rect_corners(pts)
    loss = rect.pow(2).sum()
    loss.backward()
    assert pts.grad is not None
    assert pts.grad.abs().sum() > 0


def test_adapter_makes_compliant_input_zero_energy_under_door_rule() -> None:
    """End-to-end: 32-vertex 1.5 m square -> 4 corners -> door rule (>=0.85 m) = 0."""
    t = torch.linspace(0.0, 4.0, 33, dtype=torch.float64)[:-1]
    pts = []
    for ti in t:
        seg = int(ti.item()) % 4
        f = float(ti.item()) - seg
        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        x0, y0 = corners[seg]
        x1, y1 = corners[(seg + 1) % 4]
        pts.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
    polygon = torch.tensor(pts, dtype=torch.float64) * 1.5
    rect = polygons_to_rotated_rect(polygon)  # (4, 2)
    e = door_width_energy(rect.unsqueeze(0), min_width_m=0.85)
    assert float(e) < 1e-6


def test_adapter_rejects_wrong_last_dim() -> None:
    bad = torch.zeros(4, 3, dtype=torch.float64)  # 3-d coords instead of 2
    try:
        polygons_to_rotated_rect(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for last-dim != 2")
