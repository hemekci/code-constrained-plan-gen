"""MSD dataset loader.

Mirrors the loading logic from `data/MSD/graphs.py` (extract_access_graph)
but wraps the output in our typed `PlanGraph` so downstream rule-checking
and modeling code sees a single canonical representation.

Reference: van Engelenburg et al. "MSD: A Benchmark Dataset for Floor Plan
Generation of Building Complexes", ECCV 2024.

The Kaggle download produces a parquet (or pickle) of geometry rows. The
schema we expect (verified against `data/MSD/graphs.py`):

    columns: floor_id, geom (WKT string), zoning (str), room_type (str)

If the actual on-disk schema differs after download, this module raises a
clear error pointing at this file rather than failing deep in the pipeline.
"""

from __future__ import annotations

import logging
import pickle
from itertools import combinations
from pathlib import Path
from typing import Iterator, Optional

import networkx as nx
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon

from code_module import Connectivity, PlanGraph, RoomNode

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "MSD" / "raw"

# Mirrors data/MSD/constants.py::ROOM_NAMES (full vocabulary)
ROOM_NAMES = [
    "Bedroom",
    "Livingroom",
    "Kitchen",
    "Dining",
    "Corridor",
    "Stairs",
    "Storeroom",
    "Bathroom",
    "Balcony",
    "Structure",
    "Door",
    "Entrance Door",
    "Window",
]

# Distance thresholds from MSD/graphs.py — these are in MSD's normalized units
PASSAGE_DISTANCE_THRESHOLD = 0.04
DOOR_DISTANCE_THRESHOLD = 0.05


def _resolve_table_path(raw_dir: Path) -> Path:
    """Find the geometry table file regardless of MSD's exact filename.

    Returns the first file matching common conventions; raises FileNotFoundError
    with explicit guidance if none is found.
    """
    candidates = (
        list(raw_dir.glob("*.parquet"))
        + list(raw_dir.glob("*.pkl"))
        + list(raw_dir.glob("*.pickle"))
        + list(raw_dir.glob("*.csv"))
    )
    candidates = [p for p in candidates if "geom" in p.name.lower() or "msd" in p.name.lower() or len(candidates) == 1]
    if not candidates:
        raise FileNotFoundError(
            f"No geometry table found under {raw_dir}. "
            "Expected a .parquet/.pkl/.csv with columns "
            "(floor_id, geom, zoning, room_type). After Kaggle download, "
            "verify the layout via scripts/inspect_datasets.py."
        )
    return sorted(candidates)[0]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _validate_columns(df: pd.DataFrame) -> None:
    needed = {"floor_id", "geom"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(
            f"MSD table missing required columns {missing}. Available: {sorted(df.columns)}"
        )


def _coerce_geometry(value: object) -> Polygon:
    if isinstance(value, Polygon):
        return value
    if isinstance(value, (str, bytes)):
        return wkt.loads(value)
    raise TypeError(f"Cannot coerce geometry value of type {type(value).__name__}")


def _build_graph_for_floor(
    floor_df: pd.DataFrame,
    column: str,
) -> nx.Graph:
    """Reproduce MSD/graphs.py::extract_access_graph using the room_type column.

    `column` selects which label column to treat as the node 'room_type' —
    typically `'room_type'` for the full output graph or `'zoning'` for the
    coarser zoning input graph.
    """
    if column not in floor_df.columns:
        raise KeyError(
            f"Column '{column}' not in floor frame; columns: {sorted(floor_df.columns)}"
        )

    floor_df = floor_df.assign(
        _poly=floor_df["geom"].apply(_coerce_geometry),
        _label=floor_df[column].astype(str),
    )

    # Partition by label
    is_door = floor_df["_label"] == "Door"
    is_entrance = floor_df["_label"] == "Entrance Door"
    is_room = ~(is_door | is_entrance) & (floor_df["_label"] != "Window") & (
        floor_df["_label"] != "Structure"
    ) & (floor_df["_label"] != "Wall")

    rooms = list(zip(floor_df.loc[is_room, "_poly"], floor_df.loc[is_room, "_label"], strict=True))
    doors = list(floor_df.loc[is_door, "_poly"])
    entrances = list(floor_df.loc[is_entrance, "_poly"])

    g = nx.Graph()
    for idx, (poly, label) in enumerate(rooms):
        g.add_node(
            idx,
            geometry=poly,
            room_type=label,
            centroid=(float(poly.centroid.x), float(poly.centroid.y)),
        )

    # Pairwise edges
    for (i, (poly_i, _)), (j, (poly_j, _)) in combinations(enumerate(rooms), 2):
        if poly_i.distance(poly_j) < PASSAGE_DISTANCE_THRESHOLD:
            g.add_edge(i, j, connectivity=Connectivity.PASSAGE)
            continue
        # door check
        for door in doors:
            if (
                door.distance(poly_i) < DOOR_DISTANCE_THRESHOLD
                and door.distance(poly_j) < DOOR_DISTANCE_THRESHOLD
            ):
                g.add_edge(i, j, connectivity=Connectivity.DOOR, door_geometry=door)
                break

    # Entrance edges (a room that touches an entrance door becomes reachable from outside)
    for i, (poly_i, _) in enumerate(rooms):
        for ent in entrances:
            if ent.distance(poly_i) < DOOR_DISTANCE_THRESHOLD:
                g.add_edge(i, i, connectivity=Connectivity.ENTRANCE, door_geometry=ent)
                break

    return g


def load_msd_floor(
    floor_id: str | int,
    raw_dir: Optional[Path] = None,
    column: str = "room_type",
    unit_scale_m_per_unit: float = 1.0,
) -> PlanGraph:
    """Load a single MSD floor as a PlanGraph.

    Parameters
    ----------
    floor_id
        The floor identifier present in the MSD table.
    raw_dir
        Override for `data/MSD/raw`.
    column
        Which label column to use for `room_type` (use `'zoning'` for the
        coarser input graph, `'room_type'` for the full output graph).
    unit_scale_m_per_unit
        Conversion factor from MSD's normalized coordinates to meters. The
        actual factor depends on MSD's storage convention and must be
        determined empirically once raw data is on disk.
    """
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    table_path = _resolve_table_path(raw_dir)
    df = _read_table(table_path)
    _validate_columns(df)

    floor_df = df[df["floor_id"] == floor_id].reset_index(drop=True)
    if floor_df.empty:
        raise KeyError(f"floor_id={floor_id!r} not found in {table_path}")

    g = _build_graph_for_floor(floor_df, column=column)
    return PlanGraph(
        graph=g,
        unit_scale_m_per_unit=unit_scale_m_per_unit,
        plan_id=str(floor_id),
    )


def iter_msd_floors(
    raw_dir: Optional[Path] = None,
    column: str = "room_type",
    unit_scale_m_per_unit: float = 1.0,
    limit: Optional[int] = None,
) -> Iterator[PlanGraph]:
    """Iterate over MSD floors yielding PlanGraph instances."""
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    table_path = _resolve_table_path(raw_dir)
    df = _read_table(table_path)
    _validate_columns(df)
    floor_ids = df["floor_id"].drop_duplicates()
    if limit is not None:
        floor_ids = floor_ids.head(limit)
    for floor_id in floor_ids:
        floor_df = df[df["floor_id"] == floor_id].reset_index(drop=True)
        try:
            g = _build_graph_for_floor(floor_df, column=column)
        except Exception as exc:  # noqa: BLE001 — log and skip per floor
            logger.warning("Failed to build graph for floor %s: %s", floor_id, exc)
            continue
        yield PlanGraph(
            graph=g,
            unit_scale_m_per_unit=unit_scale_m_per_unit,
            plan_id=str(floor_id),
        )


def _coerce_centroid(value: object) -> tuple[float, float]:
    """Convert MSD centroid (torch.Tensor / tuple / list) into a plain tuple."""
    if value is None:
        return (0.0, 0.0)
    if hasattr(value, "tolist"):
        as_list = value.tolist()
        if len(as_list) >= 2:
            return float(as_list[0]), float(as_list[1])
    if hasattr(value, "__iter__"):
        as_list = list(value)
        if len(as_list) >= 2:
            return float(as_list[0]), float(as_list[1])
    return (0.0, 0.0)


def _coerce_room_type(value: object) -> str:
    """Convert MSD room_type (int index or string label) into a string label."""
    if isinstance(value, str):
        return value
    try:
        idx = int(value)
        if 0 <= idx < len(ROOM_NAMES):
            return ROOM_NAMES[idx]
    except (TypeError, ValueError):
        pass
    return str(value)


def _coerce_geometry_from_coords(value: object) -> Polygon:
    """MSD pre-extracted graphs store geometry as a list of (x, y) tuples."""
    if isinstance(value, Polygon):
        return value
    if isinstance(value, list) and value and isinstance(value[0], tuple):
        return Polygon(value)
    if isinstance(value, list) and value and hasattr(value[0], "__iter__"):
        return Polygon([tuple(p) for p in value])
    raise TypeError(
        f"Cannot coerce MSD geometry of type {type(value).__name__} into Polygon"
    )


def _coerce_connectivity(value: object) -> Connectivity:
    """MSD edges store connectivity as a plain string."""
    if isinstance(value, Connectivity):
        return value
    if isinstance(value, str):
        try:
            return Connectivity(value)
        except ValueError:
            pass
    return Connectivity.PASSAGE


def load_msd_floor_pickle(pickle_path: Path | str) -> PlanGraph:
    """Load a single MSD floor from a pre-extracted networkx pickle.

    These pickles live under
    `data/MSD/raw/modified-swiss-dwellings-v2/{train,test}/graph_out/<floor_id>.pickle`.
    Coordinates are stored in meters (real Swiss building units) so we set
    `unit_scale_m_per_unit=1.0`.

    Note: pre-extracted graphs do NOT carry per-edge `door_geometry`, so the
    `DoorWidth` rule will produce empty results on these graphs. To check door
    widths, recompute the graph from the CSV table via `load_msd_floor`.
    """
    pickle_path = Path(pickle_path)
    with pickle_path.open("rb") as fh:
        raw_graph = pickle.load(fh)
    if not isinstance(raw_graph, nx.Graph):
        raise TypeError(
            f"Expected networkx.Graph, got {type(raw_graph).__name__} from {pickle_path}"
        )

    g = nx.Graph()
    for nid, attrs in raw_graph.nodes(data=True):
        try:
            polygon = _coerce_geometry_from_coords(attrs.get("geometry"))
        except TypeError as exc:
            logger.warning("Skipping node %s in %s: %s", nid, pickle_path.name, exc)
            continue
        g.add_node(
            nid,
            geometry=polygon,
            room_type=_coerce_room_type(attrs.get("room_type")),
            centroid=_coerce_centroid(attrs.get("centroid")),
        )
    for u, v, attrs in raw_graph.edges(data=True):
        if u not in g or v not in g:
            continue
        g.add_edge(u, v, connectivity=_coerce_connectivity(attrs.get("connectivity")))

    return PlanGraph(
        graph=g,
        unit_scale_m_per_unit=1.0,
        plan_id=pickle_path.stem,
    )


def iter_msd_pickle_floors(
    split: str = "train",
    raw_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Iterator[PlanGraph]:
    """Iterate pre-extracted MSD floors from `train/` or `test/` graph_out.

    Parameters
    ----------
    split
        `'train'` or `'test'`.
    limit
        Stop after this many floors (None = all).
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    pickle_dir = raw_dir / "modified-swiss-dwellings-v2" / split / "graph_out"
    if not pickle_dir.exists():
        raise FileNotFoundError(
            f"Expected pickle dir at {pickle_dir}; was the Kaggle download "
            "extracted correctly?"
        )
    files = sorted(pickle_dir.glob("*.pickle"))
    if limit is not None:
        files = files[:limit]
    for path in files:
        try:
            yield load_msd_floor_pickle(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", path.name, exc)
            continue


__all__ = [
    "DEFAULT_RAW_DIR",
    "PASSAGE_DISTANCE_THRESHOLD",
    "DOOR_DISTANCE_THRESHOLD",
    "ROOM_NAMES",
    "iter_msd_floors",
    "iter_msd_pickle_floors",
    "load_msd_floor",
    "load_msd_floor_pickle",
]
