"""CubiCasa5K dataset loader.

CubiCasa5K stores each floor plan as an SVG file (`model.svg`) inside a
per-plan folder, with semantic groups for Space (rooms), Door, Window,
and Wall. We parse the SVG, extract polygons, build the connectivity
graph by spatial proximity, and wrap into our typed `PlanGraph`.

Reference: Kalervo et al. "CubiCasa5K: A Dataset and an Improved
Multi-Task Model for Floorplan Image Analysis", SCIA 2019.

The CubiCasa SVG room class names are mapped down to the same vocabulary
used by MSD's pickle loader so downstream code (rules, jurisdictions)
sees a consistent room-type space across datasets.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations
from pathlib import Path
from typing import Iterator, Optional
from xml.etree import ElementTree as ET

import networkx as nx
from shapely.geometry import MultiPolygon, Polygon

from code_module import Connectivity, PlanGraph

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "CubiCasa5K" / "data" / "cubicasa5k"

# SVG namespace handling
SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {"svg": SVG_NS}

# CubiCasa room class -> our canonical vocabulary (matches MSD ROOM_NAMES).
# Anything not listed defaults to "Storeroom" (catch-all functional room).
CUBICASA_ROOM_MAP: dict[str, str] = {
    # Bedrooms
    "Bedroom": "Bedroom",
    "Room": "Bedroom",
    # Living
    "LivingRoom": "Livingroom",
    "Lounge": "Livingroom",
    "Den": "Livingroom",
    "RecreationRoom": "Livingroom",
    # Kitchen / dining
    "Kitchen": "Kitchen",
    "EatingArea": "Dining",
    "Dining": "Dining",
    "Pantry": "Kitchen",
    "Counter": "Kitchen",
    # Bath
    "Bath": "Bathroom",
    "Sauna": "Bathroom",
    "HotTub": "Bathroom",
    # Corridor / circulation
    "Hall": "Corridor",
    "HallWay": "Corridor",
    "Entry": "Corridor",
    "DraughtLobby": "Corridor",
    # Stairs / vertical circulation
    "StairWell": "Stairs",
    "Stairs": "Stairs",
    "Elevator": "Stairs",
    # Storage
    "Storage": "Storeroom",
    "Closet": "Storeroom",
    "DressingRoom": "Storeroom",
    "Garbage": "Storeroom",
    "Utility": "Storeroom",
    "TechnicalRoom": "Storeroom",
    # Outdoor
    "Outdoor": "Balcony",
    "Garage": "Storeroom",
    "CarPort": "Storeroom",
}

# Distance thresholds in image-coordinate units; calibrate per scale
DEFAULT_PASSAGE_THRESHOLD = 1.0  # pixels
DEFAULT_DOOR_THRESHOLD = 2.0  # pixels


# ---------------------------------------------------------------------------
# SVG path / polygon parsing
# ---------------------------------------------------------------------------

_PATH_TOKEN_RE = re.compile(r"([MLZmlz])|(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")


def _parse_path_d(d: str) -> list[tuple[float, float]]:
    """Parse a simple SVG path 'd' attribute into a sequence of (x, y) points.

    Handles ``M``, ``L`` (and lowercase relative ``m``, ``l``) and ``Z``.
    Curved or arc commands are not used in CubiCasa room polygons.
    """
    points: list[tuple[float, float]] = []
    cmd: Optional[str] = None
    numbers: list[float] = []
    cursor = (0.0, 0.0)

    def flush() -> None:
        nonlocal cursor, numbers, cmd
        if cmd is None:
            return
        if cmd in ("M", "L"):
            for i in range(0, len(numbers) - 1, 2):
                cursor = (numbers[i], numbers[i + 1])
                points.append(cursor)
        elif cmd in ("m", "l"):
            for i in range(0, len(numbers) - 1, 2):
                cursor = (cursor[0] + numbers[i], cursor[1] + numbers[i + 1])
                points.append(cursor)
        # Z / z close the path (no-op for our extraction)
        numbers = []

    for tok in _PATH_TOKEN_RE.finditer(d):
        letter, num = tok.group(1), tok.group(2)
        if letter is not None:
            flush()
            cmd = letter
        elif num is not None:
            numbers.append(float(num))
    flush()
    return points


def _parse_polygon_points(value: str) -> list[tuple[float, float]]:
    """Parse a `<polygon points="x1,y1 x2,y2 ...">` attribute."""
    pts: list[tuple[float, float]] = []
    for chunk in value.replace(",", " ").split():
        try:
            pts.append((float(chunk), float(pts[-1][1] if pts else 0.0)))
        except ValueError:
            continue
    # Re-parse properly (above hack drops y values); use a robust 2-stride parse instead
    nums = [float(x) for x in value.replace(",", " ").split() if x]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _element_to_polygon(elem: ET.Element) -> Optional[Polygon]:
    """Best-effort extraction of a single Polygon from an SVG element subtree."""
    tag = elem.tag.split("}", 1)[-1]
    if tag == "path":
        d = elem.get("d", "")
        pts = _parse_path_d(d)
    elif tag == "polygon":
        pts = _parse_polygon_points(elem.get("points", ""))
    elif tag == "rect":
        x = float(elem.get("x", "0"))
        y = float(elem.get("y", "0"))
        w = float(elem.get("width", "0"))
        h = float(elem.get("height", "0"))
        if w <= 0 or h <= 0:
            return None
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    else:
        # Recurse into children if the element is a group
        for child in elem:
            poly = _element_to_polygon(child)
            if poly is not None:
                return poly
        return None

    if len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        # buffer(0) on a self-intersecting polygon can produce a MultiPolygon;
        # take the largest connected component as the room footprint.
        if isinstance(poly, MultiPolygon):
            if poly.is_empty:
                return None
            poly = max(poly.geoms, key=lambda p: p.area)
        if poly.is_empty or poly.area < 1.0:
            return None
        return poly
    except Exception as exc:  # noqa: BLE001
        logger.debug("Polygon parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CubiCasa floor loader
# ---------------------------------------------------------------------------


def _classify_room(class_attr: str) -> Optional[str]:
    """Map a 'Space <RoomType>' SVG class string into our vocabulary."""
    parts = class_attr.split()
    if not parts or parts[0] != "Space":
        return None
    if len(parts) < 2:
        return None
    cubicasa_type = parts[1]
    return CUBICASA_ROOM_MAP.get(cubicasa_type, "Storeroom")


def load_cubicasa_floor(
    svg_path: Path | str,
    *,
    unit_scale_m_per_unit: float = 0.01,
    passage_distance_threshold: float = DEFAULT_PASSAGE_THRESHOLD,
    door_distance_threshold: float = DEFAULT_DOOR_THRESHOLD,
) -> PlanGraph:
    """Load one CubiCasa5K plan from its `model.svg` into a `PlanGraph`.

    Parameters
    ----------
    svg_path
        Path to a `model.svg` file (or to its enclosing folder).
    unit_scale_m_per_unit
        Conversion from SVG units (pixels in CubiCasa) to meters. The
        CubiCasa documentation uses cm-per-pixel scaling that varies per
        plan; ``0.01`` (1 cm per SVG unit) is a reasonable starting default
        for the **scaled** images. Calibrate empirically by inspecting room
        sizes against expected residential dimensions.
    """
    svg_path = Path(svg_path)
    if svg_path.is_dir():
        svg_path = svg_path / "model.svg"

    tree = ET.parse(svg_path)
    root = tree.getroot()

    rooms: list[tuple[Polygon, str]] = []
    doors: list[Polygon] = []
    windows: list[Polygon] = []  # not used directly but parsed for completeness

    # Walk every group in the document; SVG namespace may or may not be present
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag != "g":
            continue
        gid = elem.get("id", "")
        gclass = elem.get("class", "")

        if gid == "Door":
            poly = _element_to_polygon(elem)
            if poly is not None:
                doors.append(poly)
            continue
        if gid == "Window":
            poly = _element_to_polygon(elem)
            if poly is not None:
                windows.append(poly)
            continue

        if gclass.startswith("Space "):
            room_type = _classify_room(gclass)
            if room_type is None:
                continue
            poly = _element_to_polygon(elem)
            if poly is not None:
                rooms.append((poly, room_type))

    # Build graph
    g = nx.Graph()
    for idx, (poly, rtype) in enumerate(rooms):
        g.add_node(
            idx,
            geometry=poly,
            room_type=rtype,
            centroid=(float(poly.centroid.x), float(poly.centroid.y)),
        )

    for (i, (pi, _)), (j, (pj, _)) in combinations(enumerate(rooms), 2):
        if pi.distance(pj) < passage_distance_threshold:
            g.add_edge(i, j, connectivity=Connectivity.PASSAGE)
            continue
        # Door connection
        for door in doors:
            if (
                door.distance(pi) < door_distance_threshold
                and door.distance(pj) < door_distance_threshold
            ):
                g.add_edge(
                    i,
                    j,
                    connectivity=Connectivity.DOOR,
                    door_geometry=door,
                )
                break

    # Entrance heuristic: a real entrance is a Door element on the building
    # perimeter. Find doors whose centroid is within `entrance_boundary_margin`
    # of the global plan bbox, then mark whichever room is closest to each
    # such door as having an Entrance self-loop. This is much tighter than
    # the previous "any room touching the bbox" heuristic, which fired on
    # every perimeter-adjacent bedroom.
    if rooms:
        all_pts = [pt for r, _ in rooms for pt in r.exterior.coords]
        if all_pts:
            xs = [x for x, _ in all_pts]
            ys = [y for _, y in all_pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            entrance_boundary_margin = max(
                door_distance_threshold * 2.0,
                0.01 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]),
            )

            entrance_rooms: set[int] = set()
            for door in doors:
                cx, cy = door.centroid.x, door.centroid.y
                on_perimeter = (
                    abs(cx - bbox[0]) < entrance_boundary_margin
                    or abs(cy - bbox[1]) < entrance_boundary_margin
                    or abs(cx - bbox[2]) < entrance_boundary_margin
                    or abs(cy - bbox[3]) < entrance_boundary_margin
                )
                if not on_perimeter:
                    continue
                # Attach the door to the closest room (the building-side room
                # of an exterior door — by construction the only adjacent room).
                best_idx, best_d = None, float("inf")
                for idx, (poly, _) in enumerate(rooms):
                    d = door.distance(poly)
                    if d < best_d:
                        best_idx, best_d = idx, d
                if best_idx is not None and best_d < door_distance_threshold:
                    entrance_rooms.add(best_idx)

            # Fallback: if no perimeter doors were detected (CubiCasa SVGs
            # vary), fall back to a single boundary-touching room so egress
            # rules at least have one exit to pin to.
            if not entrance_rooms:
                margin = 1.0
                for idx, (poly, _) in enumerate(rooms):
                    px_min, py_min, px_max, py_max = poly.bounds
                    touches_boundary = (
                        abs(px_min - bbox[0]) < margin
                        or abs(py_min - bbox[1]) < margin
                        or abs(px_max - bbox[2]) < margin
                        or abs(py_max - bbox[3]) < margin
                    )
                    if touches_boundary:
                        entrance_rooms.add(idx)
                        break

            for idx in entrance_rooms:
                g.add_edge(idx, idx, connectivity=Connectivity.ENTRANCE)

    return PlanGraph(
        graph=g,
        unit_scale_m_per_unit=unit_scale_m_per_unit,
        plan_id=svg_path.parent.name if svg_path.parent.name else svg_path.stem,
    )


def iter_cubicasa_floors(
    raw_dir: Optional[Path] = None,
    split_file: str = "train.txt",
    *,
    unit_scale_m_per_unit: float = 0.01,
    limit: Optional[int] = None,
) -> Iterator[PlanGraph]:
    """Iterate CubiCasa5K plans listed in a split file (train/val/test.txt)."""
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    split_path = Path(raw_dir) / split_file
    if not split_path.exists():
        raise FileNotFoundError(
            f"Expected split file {split_path}. After Zenodo extract, the layout is:\n"
            f"  data/CubiCasa5K/data/cubicasa5k/\n"
            f"    high_quality/  high_quality_architectural/  colorful/\n"
            f"    train.txt  val.txt  test.txt"
        )
    folders = [
        line.strip()
        for line in split_path.read_text().splitlines()
        if line.strip()
    ]
    if limit is not None:
        folders = folders[:limit]
    for rel in folders:
        plan_dir = Path(raw_dir) / rel.lstrip("/")
        svg_path = plan_dir / "model.svg"
        if not svg_path.exists():
            logger.warning("Missing SVG: %s", svg_path)
            continue
        try:
            yield load_cubicasa_floor(
                svg_path, unit_scale_m_per_unit=unit_scale_m_per_unit
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", svg_path, exc)
            continue


__all__ = [
    "CUBICASA_ROOM_MAP",
    "DEFAULT_DOOR_THRESHOLD",
    "DEFAULT_PASSAGE_THRESHOLD",
    "DEFAULT_RAW_DIR",
    "iter_cubicasa_floors",
    "load_cubicasa_floor",
]
