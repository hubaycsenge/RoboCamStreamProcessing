"""The 2D occupancy grid: decoding, patching, and the world-to-cell arithmetic.

This is the shared object in the middle of the system diagram.  The robot's SLAM
draws it, the robot's Nav2 drives on it, the server's Compare diffs a
reconstruction against it, and Compare's output goes back into it.  Four
consumers, one layout — so the layout is defined here once and nothing else is
allowed an opinion about it.

The layout is ``nav_msgs/OccupancyGrid``'s, unchanged
------------------------------------------------------
Cells are int8: ``-1`` unknown, ``0..100`` percent probability of occupancy.
Storage is row-major with row 0 at the origin, so cell ``(col, row)`` is at
``data[row * width + col]``, and its centre in world coordinates is

    x = origin_x + (col + 0.5) * resolution
    y = origin_y + (row + 0.5) * resolution

rotated by ``origin_yaw`` about the origin when that is nonzero.  Keeping ROS's
exact convention is not deference to ROS — it is that the robot already holds
the grid in this form, and every re-origining on the way through a link is one
more chance to flip a map north for south and produce something that looks like
a map, navigates like a map, and is a mirror image.

Three cell classes, not two
---------------------------
Unknown is a class, not a missing value.  A robot that treats unknown as free
drives into rooms it has never seen; one that treats it as occupied never
explores at all.  Everything here therefore carries unknown through
untouched — :func:`classify` returns three masks, and the merge in
:func:`apply_patch` never lets a patch write unknown over a cell that has been
observed.

The threshold pair (``occupied_min``, ``free_max``) travels with the grid rather
than living in this server's config, because it is the *robot's* costmap tuning.
A server that hardcoded ROS's 65/25 while the robot ran at 50/40 would disagree
with it about which cells are obstacles, and that disagreement would surface only
as a comparison finding differences along every wall.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import wire

#: Refuse a grid bigger than this many cells before allocating anything.  40
#: million cells is a 1 km square at 5 cm — far past anything this robot maps,
#: and small enough that a corrupt width/height cannot turn into a GB.
MAX_CELLS = 40_000_000

#: Refuse a resolution outside this range.  A grid at 1 mm or at 10 m is not a
#: map of a room; both are what a units error looks like.
MIN_RESOLUTION_M = 0.001
MAX_RESOLUTION_M = 10.0


class MapError(Exception):
    """Raised when an occupancy-grid header or payload does not parse."""


@dataclass
class Grid:
    """One decoded occupancy grid, or one patch of one.

    ``cells`` is int8 of shape ``(height, width)`` — row-major, row 0 at the
    origin, which is numpy's natural indexing for this layout: ``cells[row,
    col]``.  It is *not* transposed to (x, y) on the way in.  Transposing would
    make the indexing read more naturally in a couple of places here and would
    silently disagree with every other consumer of the same buffer, which is a
    bad trade.
    """

    cells: np.ndarray             # int8, (height, width), -1 unknown, 0..100
    resolution: float             # metres per cell
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # x, y, yaw of cell (0,0)
    frame: str = "map"
    map_id: str = ""
    seq: int = -1
    # The robot's own thresholds, carried with the grid.  See the module
    # docstring for why these are not the server's to choose.
    occupied_min: int = 65
    free_max: int = 25
    # Offset of this grid within the full one, in cells.  (0, 0) for a full grid.
    x0: int = 0
    y0: int = 0
    full: bool = True
    source: str = ""
    # Server monotonic clock, ns, when it came off the socket.
    recv_ts_ns: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    @property
    def size(self) -> int:
        return int(self.cells.size)

    def world_to_cell(self, x, y):
        """World metres to (col, row) indices.  Vectorised; no bounds check.

        Returns integer arrays, which may be outside the grid — deliberately.
        Clamping here would pile every out-of-map point onto the border cells and
        build a wall around the map out of things that were never in it; the
        caller masks with :meth:`in_bounds` instead.
        """
        ox, oy, oyaw = self.origin
        dx = np.asarray(x, dtype=np.float64) - ox
        dy = np.asarray(y, dtype=np.float64) - oy
        if oyaw:
            # Rotate world offsets into the grid's own axes.  Nearly every grid
            # this robot produces has yaw 0, but a SLAM node that starts the map
            # aligned to the robot rather than to the world does not, and the
            # failure is a map rotated by exactly the robot's initial heading.
            c, s = math.cos(-oyaw), math.sin(-oyaw)
            dx, dy = c * dx - s * dy, s * dx + c * dy
        col = np.floor(dx / self.resolution).astype(np.int64)
        row = np.floor(dy / self.resolution).astype(np.int64)
        return col, row

    def cell_to_world(self, col, row):
        """(col, row) indices to the world metres of each cell's *centre*."""
        ox, oy, oyaw = self.origin
        dx = (np.asarray(col, dtype=np.float64) + 0.5) * self.resolution
        dy = (np.asarray(row, dtype=np.float64) + 0.5) * self.resolution
        if oyaw:
            c, s = math.cos(oyaw), math.sin(oyaw)
            dx, dy = c * dx - s * dy, s * dx + c * dy
        return dx + ox, dy + oy

    def in_bounds(self, col, row) -> np.ndarray:
        col = np.asarray(col)
        row = np.asarray(row)
        return (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)

    def classify(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boolean masks for (occupied, free, unknown), in that order.

        Unknown is everything negative rather than everything equal to -1: a
        grid that arrives with -2 in it is malformed, and reading that as
        "probability -2" would be worse than reading it as "not observed".
        """
        unknown = self.cells < 0
        occupied = self.cells >= self.occupied_min
        free = (~unknown) & (self.cells <= self.free_max)
        return occupied, free, unknown

    def counts(self) -> Dict[str, int]:
        occupied, free, unknown = self.classify()
        return {
            "occupied": int(occupied.sum()),
            "free": int(free.sum()),
            "unknown": int(unknown.sum()),
            # Cells between free_max and occupied_min: observed, and neither.
            "uncertain": int(self.size - occupied.sum() - free.sum() - unknown.sum()),
        }

    def extent_m(self) -> Tuple[float, float]:
        return (self.width * self.resolution, self.height * self.resolution)

    def copy(self) -> "Grid":
        clone = Grid(
            cells=self.cells.copy(), resolution=self.resolution, origin=self.origin,
            frame=self.frame, map_id=self.map_id, seq=self.seq,
            occupied_min=self.occupied_min, free_max=self.free_max,
            x0=self.x0, y0=self.y0, full=self.full, source=self.source,
            recv_ts_ns=self.recv_ts_ns,
        )
        clone.summary = dict(self.summary)
        return clone


def decode_map(header: Dict[str, Any], payload: bytes, recv_ts_ns: int = 0,
               max_cells: int = MAX_CELLS) -> Grid:
    """Turn one ``map`` message into a :class:`Grid`.

    Every size is checked *before* the allocation it would drive, which is the
    only ordering that helps: a corrupt width field is discovered by trying to
    allocate what it asks for, and by then the damage is done.
    """
    try:
        width = int(header.get("width", 0))
        height = int(header.get("height", 0))
        resolution = float(header.get("resolution", 0.0))
    except (TypeError, ValueError) as exc:
        raise MapError(f"grid geometry is not numeric: {exc}") from exc

    if width <= 0 or height <= 0:
        raise MapError(f"grid is {width}x{height}; both dimensions must be positive")
    if width * height > max_cells:
        raise MapError(
            f"grid is {width}x{height} = {width * height} cells, over the "
            f"{max_cells} cell limit"
        )
    if not (MIN_RESOLUTION_M <= resolution <= MAX_RESOLUTION_M):
        raise MapError(
            f"resolution {resolution} m/cell is outside "
            f"[{MIN_RESOLUTION_M}, {MAX_RESOLUTION_M}]; check the units"
        )

    encoding = str(header.get("encoding", wire.MAP_ENC_I8))
    if encoding not in wire.SUPPORTED_MAP_ENCODINGS:
        raise MapError(
            f"unsupported map encoding {encoding!r}; supported: "
            f"{', '.join(wire.SUPPORTED_MAP_ENCODINGS)}"
        )

    raw = payload
    if encoding == wire.MAP_ENC_I8_ZLIB:
        try:
            # Bounded so that a hostile or corrupt payload cannot inflate into
            # memory: one byte per declared cell is exactly what a valid one
            # produces, and zlib stops there.
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(payload, width * height + 1)
            if len(raw) > width * height:
                raise MapError(
                    f"compressed payload expands past the declared {width}x{height} grid"
                )
        except zlib.error as exc:
            raise MapError(f"zlib payload did not decompress: {exc}") from exc

    if len(raw) != width * height:
        raise MapError(
            f"payload is {len(raw)} bytes for a {width}x{height} = "
            f"{width * height} cell grid"
        )

    cells = np.frombuffer(raw, dtype=np.int8).reshape(height, width)

    origin_raw = header.get("origin") or (0.0, 0.0, 0.0)
    try:
        ox, oy, oyaw = (float(v) for v in list(origin_raw)[:3])
    except (TypeError, ValueError) as exc:
        raise MapError(f"origin is not three numbers: {exc}") from exc
    if not all(math.isfinite(v) for v in (ox, oy, oyaw)):
        raise MapError(f"origin ({ox}, {oy}, {oyaw}) is not finite")

    occupied_min = int(header.get("occupied_min", 65))
    free_max = int(header.get("free_max", 25))
    if not 0 <= free_max < occupied_min <= 100:
        raise MapError(
            f"thresholds free_max={free_max} occupied_min={occupied_min} do not "
            "satisfy 0 <= free_max < occupied_min <= 100"
        )

    return Grid(
        # A copy, not the read-only view frombuffer hands back: the server holds
        # this grid across frames and patches it in place.
        cells=cells.copy(),
        resolution=resolution,
        origin=(ox, oy, oyaw),
        frame=str(header.get("frame", "map") or "map"),
        map_id=str(header.get("map_id", "")),
        seq=int(header.get("seq", -1)),
        occupied_min=occupied_min,
        free_max=free_max,
        x0=int(header.get("x0", 0)),
        y0=int(header.get("y0", 0)),
        full=bool(header.get("full", True)),
        source=str(header.get("source", "")),
        recv_ts_ns=recv_ts_ns,
    )


def encode_cells(cells: np.ndarray, encoding: str = wire.MAP_ENC_I8_ZLIB) -> bytes:
    """Pack an int8 grid for the wire.

    ``level=6`` rather than 9: a mostly-unknown grid compresses to within a few
    percent of the same size either way, and 9 costs several times the CPU on the
    IO thread of a server whose whole design is that the IO thread does nothing
    slow.
    """
    raw = np.ascontiguousarray(cells, dtype=np.int8).tobytes()
    if encoding == wire.MAP_ENC_I8_ZLIB:
        return zlib.compress(raw, 6)
    if encoding == wire.MAP_ENC_I8:
        return raw
    raise MapError(f"unsupported map encoding {encoding!r}")


def apply_patch(grid: Grid, patch: Grid, merge: str = wire.MAP_MERGE_MAX) -> int:
    """Merge ``patch`` into ``grid`` in place; return the number of cells changed.

    Both ends run this — the server to maintain its copy of the robot's map from
    incremental uploads, the robot to apply what Compare sends back — which is
    the reason it is written once here rather than twice, differently.

    ``max`` merging is the important case and its rule is: a patch may raise a
    cell's occupancy and may fill in an unknown one, but may never lower an
    observed cell or return one to unknown.  That asymmetry is what makes the
    server safe to be wrong.  Its evidence is a monocular reconstruction, which
    can invent a surface but which — when it does — invents an obstacle, and an
    obstacle the robot drives around costs a detour.  The opposite policy would
    let a missing surface clear a wall the robot's own scanner saw, and that
    costs a collision.
    """
    if merge not in wire.SUPPORTED_MAP_MERGES:
        raise MapError(f"unsupported merge mode {merge!r}")

    x0, y0 = patch.x0, patch.y0
    x1, y1 = x0 + patch.width, y0 + patch.height
    # Clip to the destination.  A patch hanging off the edge is normal when the
    # robot's map has grown since the server's copy was taken, and dropping the
    # whole patch for it would throw away the part that does fit.
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(grid.width, x1), min(grid.height, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return 0

    target = grid.cells[dy0:dy1, dx0:dx1]
    source = patch.cells[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]

    if merge == wire.MAP_MERGE_REPLACE:
        changed = int(np.count_nonzero(target != source))
        target[:] = source
        return changed

    # MAP_MERGE_MAX.  Unknown (-1) in the source says "no opinion" and must not
    # win a numeric maximum against an observed 0, so it is masked out first
    # rather than left to the arithmetic — which would work by accident here,
    # since -1 < 0, and stop working the moment unknown is spelled differently.
    has_opinion = source >= 0
    merged = np.where(has_opinion & (source > target), source, target)
    changed = int(np.count_nonzero(merged != target))
    target[:] = merged
    return changed


def patch_from_diff(base: Grid, updated_cells: np.ndarray,
                    unknown: int = wire.MAP_UNKNOWN) -> Optional[Grid]:
    """The smallest patch carrying every cell where ``updated_cells`` differs.

    Returns None when nothing differs — which is the common case and the one
    worth making cheap, because it is what a comparison that found nothing
    produces and there is no reason for that to cost a message.

    The patch is the bounding box of the changes, with unchanged cells inside it
    written as unknown.  A bounding box rather than a list of cells because the
    changes cluster (a table top is a blob, not a scatter) and because the
    receiving end already knows how to merge a rectangle; the unknowns inside it
    cost a byte each before compression and nothing after it.
    """
    if updated_cells.shape != base.cells.shape:
        raise MapError(
            f"updated cells are {updated_cells.shape}, base grid is {base.cells.shape}"
        )
    diff = updated_cells != base.cells
    if not diff.any():
        return None

    rows = np.flatnonzero(diff.any(axis=1))
    cols = np.flatnonzero(diff.any(axis=0))
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1

    window = np.full((r1 - r0, c1 - c0), unknown, dtype=np.int8)
    sub_diff = diff[r0:r1, c0:c1]
    window[sub_diff] = updated_cells[r0:r1, c0:c1][sub_diff]

    return Grid(
        cells=window,
        resolution=base.resolution,
        origin=base.origin,
        frame=base.frame,
        map_id=base.map_id,
        occupied_min=base.occupied_min,
        free_max=base.free_max,
        x0=c0 + base.x0,
        y0=r0 + base.y0,
        full=False,
        source="compare",
    )


def analyse(grid: Grid) -> Dict[str, Any]:
    """The summary returned with every uploaded grid.

    ``explored_fraction`` is the number the robot's exploration policy actually
    wants and the one nobody can compute from the other fields: it is the share
    of the grid that has been observed at all, which is what "is this map
    finished" means and is why the ``Exit?`` box exists.
    """
    counts = grid.counts()
    w, h = grid.extent_m()
    observed = grid.size - counts["unknown"]
    return {
        "width": grid.width,
        "height": grid.height,
        "resolution": grid.resolution,
        "extent_m": [round(w, 2), round(h, 2)],
        "origin": [round(v, 4) for v in grid.origin],
        "frame": grid.frame,
        "map_id": grid.map_id,
        "full": grid.full,
        **counts,
        "explored_fraction": round(observed / grid.size, 4) if grid.size else 0.0,
        "occupied_fraction": round(counts["occupied"] / observed, 4) if observed else 0.0,
        "thresholds": {"free_max": grid.free_max, "occupied_min": grid.occupied_min},
    }
