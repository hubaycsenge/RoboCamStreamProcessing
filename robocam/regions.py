"""
Turning a cell-by-cell comparison into regions the robot can act on.

:mod:`robocam.compare` answers per cell: this one the cloud calls occupied and
the map calls free, that one the map has and the cloud never saw.  The robot
cannot use that.  ``mecanumbot_map_agreement`` wants *places* — a centre, a
radius, what kind of disagreement it is and how tall the structure there is —
because what it does with one is drive to it, mark it lethal, or ignore it, and
none of those are things you do to a cell.

So this module clusters.  Three decisions in it are worth stating, because each
one is the difference between a list the robot can use and a list it cannot:

**Blocks, not cells.**  Clustering raw 5 cm cells would produce hundreds of
regions per frame, most of them one cell of reconstruction noise, and the robot
would spend T1 driving to specks.  Cells are first pooled into blocks of
``block_m`` (0.5 m by default, about the footprint of the thing that would be
worth going to look at) and the clustering runs on blocks.  A block has to hold
``min_cells_per_block`` disagreeing cells before it counts at all, which is the
same speck filter ``compare.min_new_cells`` applies to the patch.

**One kind per region.**  ``cloud_only`` and ``map_only`` are not two flavours
of the same thing — one is a table top the lidar cannot see and the other is
glass the camera cannot reconstruct, and the robot's responses are opposite
(make it lethal; leave it alone).  Clustering them together would produce
regions with a majority kind and a wrong answer for the rest, so components are
grown within a kind and never across.

**Height is the payload.**  A ``cloud_only`` region is only interesting because
it has a z, and the z decides whether it is an overhang to drive under, an
obstacle to avoid or a threshold strip to roll over.  It is carried as the
**maximum** height in the region, not the mean: a region containing a table top
and its shadow is a table, and averaging it produces a height that describes
neither.  ``nan`` means the region has no structure to measure (``map_only``,
``unobserved``), and the robot is required to treat an unmeasurable height as
blocking rather than as absent.

Nothing here uses scipy: the cluster runs on the worker thread behind the model,
and the dependency is not worth a connected-components call over a grid this
small.  The flood fill is over blocks, of which there are a few hundred.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .occupancy import Grid

#: The four kinds, matching ``MapCloudAgreement.uncertain_kinds`` exactly.  They
#: are strings on the wire because the robot switches on them and a number would
#: make every log line a lookup.
KIND_CLOUD_ONLY = "cloud_only"
KIND_MAP_ONLY = "map_only"
KIND_UNOBSERVED = "unobserved"
KIND_DISAGREEMENT = "disagreement"

KINDS = (KIND_CLOUD_ONLY, KIND_MAP_ONLY, KIND_UNOBSERVED, KIND_DISAGREEMENT)


def cell_heights(
    points_map: np.ndarray,
    grid: Grid,
    *,
    z_min: float = 0.05,
    z_max: float = 1.6,
) -> np.ndarray:
    """Per-cell maximum height of the cloud, ``nan`` where the cloud has none.

    The *maximum* because the question the height answers is "would the robot
    hit this", and the top of a surface is what it would hit.  A table's
    underside is in the cloud too, and a mean over the two puts the table at
    knee height, which is neither a keepout nor an overhang.

    Same slice as :func:`robocam.compare.occupancy_from_cloud`, so a cell that
    is occupied there always has a height here.
    """
    heights = np.full(grid.cells.shape, np.nan, dtype=np.float32)
    if points_map.size == 0:
        return heights

    z = points_map[:, 2]
    in_slice = (z >= z_min) & (z <= z_max)
    if not in_slice.any():
        return heights

    sliced = points_map[in_slice]
    col, row = grid.world_to_cell(sliced[:, 0], sliced[:, 1])
    keep = grid.in_bounds(col, row)
    if not keep.any():
        return heights

    flat = row[keep] * grid.width + col[keep]
    # -inf as the identity for a running maximum, so a cell with points always
    # beats the "no points" sentinel however low its surface is -- a threshold
    # strip at z = 0.06 is structure, and starting from 0.0 would hide it.
    acc = np.full(grid.size, -np.inf, dtype=np.float32)
    np.maximum.at(acc, flat, sliced[keep, 2].astype(np.float32))
    seen = np.isfinite(acc)
    heights.reshape(-1)[seen] = acc[seen]
    return heights


def coverage(
    cloud_occupied: np.ndarray,
    grid: Grid,
) -> Dict[str, float]:
    """How much of each source the other one accounts for.

    Two numbers that sound alike and answer opposite questions:

    ``grid_coverage``  of the map's known-free floor, how much did the cloud
                       actually see?  **This is the T1 completeness number.**
                       A robot can drive every corridor, produce a perfect 2D
                       map, and hold a cloud of one wall — and nothing in the 2D
                       data says so.  This does.
    ``cloud_coverage`` of what the cloud reconstructed, how much falls inside
                       the map at all?  A low value with a healthy
                       ``grid_coverage`` means the cloud is reconstructing
                       something outside the mapped region: usually the far side
                       of a doorway, sometimes a placement that is drifting out
                       of the room.

    ``grid_coverage`` is measured over **free** cells rather than all known
    cells because the cloud is not expected to have points inside a wall.
    Counting occupied cells in the denominator would make a well-mapped room
    with thick walls score lower than a badly mapped one, which is backwards.
    """
    occupied_map, free_map, _ = grid.classify()
    free_cells = int(free_map.sum())
    cloud_cells = int(cloud_occupied.sum())

    # "Seen" means the cloud put a surface in the column above that floor cell.
    # It is a weak test on purpose: this asks whether the reconstruction reached
    # here at all, not whether it reconstructed it well.
    seen_free = int((cloud_occupied & free_map).sum())

    in_known = int((cloud_occupied & (free_map | occupied_map)).sum())

    return {
        "grid_coverage": round(seen_free / free_cells, 4) if free_cells else 0.0,
        "cloud_coverage": round(in_known / cloud_cells, 4) if cloud_cells else 0.0,
        "free_cells": free_cells,
    }


def classify_cells(
    cloud_occupied: np.ndarray,
    grid: Grid,
) -> Dict[str, np.ndarray]:
    """Split the disagreement into the four kinds, as boolean masks.

    ``unobserved`` is deliberately *not* "every cell the cloud missed".  That
    would be most of the map on every frame — a camera sees one direction and
    the grid is a room — and a region list dominated by "you have not looked
    behind you" is one the robot learns to ignore.  It is restricted to floor
    the map has swept free and the cloud has no opinion about, which is the
    thing a revisit can actually fix.
    """
    occupied_map, free_map, unknown_map = grid.classify()

    cloud_only = cloud_occupied & free_map
    map_only = occupied_map & ~cloud_occupied
    unobserved = free_map & ~cloud_occupied
    # Structure over cells the map never classified: the cloud is the only
    # source here, so it is news rather than a conflict, but it is also not a
    # place to go and look -- the robot has simply not mapped it yet.
    disagreement = cloud_occupied & unknown_map

    return {
        KIND_CLOUD_ONLY: cloud_only,
        KIND_MAP_ONLY: map_only,
        KIND_UNOBSERVED: unobserved,
        KIND_DISAGREEMENT: disagreement,
    }


def _components(blocks: np.ndarray) -> List[List[Tuple[int, int]]]:
    """4-connected components of a boolean block grid, as lists of (row, col).

    Iterative rather than recursive: a corridor of unobserved floor is a single
    component hundreds of blocks long, and Python's recursion limit is not a
    property the region list should depend on.
    """
    if not blocks.any():
        return []
    seen = np.zeros(blocks.shape, dtype=bool)
    out: List[List[Tuple[int, int]]] = []
    rows, cols = blocks.shape
    for r0, c0 in zip(*np.nonzero(blocks)):
        if seen[r0, c0]:
            continue
        comp: List[Tuple[int, int]] = []
        queue = deque([(int(r0), int(c0))])
        seen[r0, c0] = True
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and blocks[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        out.append(comp)
    return out


def _pool(mask: np.ndarray, block: int) -> np.ndarray:
    """Count set cells per block.  Pads to a whole number of blocks with zeros."""
    rows, cols = mask.shape
    pr = (-rows) % block
    pc = (-cols) % block
    if pr or pc:
        mask = np.pad(mask, ((0, pr), (0, pc)), constant_values=False)
    h, w = mask.shape
    return mask.reshape(h // block, block, w // block, block).sum(axis=(1, 3))


def _pool_max(values: np.ndarray, block: int) -> np.ndarray:
    """Maximum per block, ignoring ``nan``.  ``nan`` where a block has none."""
    rows, cols = values.shape
    pr = (-rows) % block
    pc = (-cols) % block
    if pr or pc:
        values = np.pad(values, ((0, pr), (0, pc)), constant_values=np.nan)
    h, w = values.shape
    reshaped = values.reshape(h // block, block, w // block, block)
    with np.errstate(all="ignore"):
        # all-nan blocks warn under np.nanmax; they are the normal case here.
        out = np.where(
            np.isnan(reshaped).all(axis=(1, 3)),
            np.nan,
            np.nanmax(np.where(np.isnan(reshaped), -np.inf, reshaped), axis=(1, 3)),
        )
    return out.astype(np.float32)


def extract_regions(
    cloud_occupied: np.ndarray,
    grid: Grid,
    heights: Optional[np.ndarray] = None,
    *,
    block_m: float = 0.5,
    min_cells_per_block: int = 3,
    max_regions: int = 32,
    kinds: Tuple[str, ...] = KINDS,
) -> List[Dict[str, Any]]:
    """Cluster the disagreement into regions, best first.

    Returns dicts shaped exactly like the parallel arrays of
    ``mecanumbot_msgs/MapCloudAgreement``: ``x``, ``y`` (map frame, metres),
    ``radius``, ``kind``, ``height``, ``score``, plus ``cells`` for the log.

    ``score`` is what the robot sorts by and it is deliberately simple —
    the region's size relative to the largest of its kind, so "worth going to"
    means "there is a lot of it".  Anything cleverer would be the server making
    a decision about the robot's time, and the robot is what knows where it is
    and what it has already visited.  It ranks; the robot chooses.

    ``max_regions`` caps the list because it rides in an announcement on the
    same socket as the frames.  The cap drops the *lowest* scoring, so a long
    tail of specks is what is lost rather than the table in the middle of the
    room.
    """
    block = max(1, int(round(block_m / grid.resolution)))
    masks = classify_cells(cloud_occupied, grid)
    if heights is None:
        heights = np.full(grid.cells.shape, np.nan, dtype=np.float32)
    pooled_heights = _pool_max(heights, block)

    regions: List[Dict[str, Any]] = []
    for kind in kinds:
        mask = masks.get(kind)
        if mask is None or not mask.any():
            continue
        counts = _pool(mask, block)
        blocks = counts >= min_cells_per_block
        for comp in _components(blocks):
            rows = np.array([r for r, _ in comp])
            cols = np.array([c for _, c in comp])
            cells = int(counts[rows, cols].sum())

            # Centre in world coordinates: the mean of the component's block
            # centres, weighted by how many disagreeing cells each holds, so an
            # L-shaped region reports the elbow rather than the empty middle of
            # its bounding box.
            weights = counts[rows, cols].astype(np.float64)
            cx_cell = float((cols * block + block / 2.0) @ weights / weights.sum())
            cy_cell = float((rows * block + block / 2.0) @ weights / weights.sum())
            # cell_to_world adds half a cell of its own; cx_cell is already an
            # absolute cell coordinate at the block's centre, so cancel it.
            x, y = grid.cell_to_world(cx_cell - 0.5, cy_cell - 0.5)

            # Radius from the extent, not from the area: the robot uses it to
            # decide where to stand, and a long thin region (a wall edge) needs
            # the distance to its far end, not the radius of a disc of equal
            # area.
            dx = (cols * block + block / 2.0) - cx_cell
            dy = (rows * block + block / 2.0) - cy_cell
            radius = float(np.sqrt(dx * dx + dy * dy).max() + block / 2.0) * grid.resolution

            h = pooled_heights[rows, cols]
            height = float(np.nanmax(h)) if np.isfinite(h).any() else float("nan")

            regions.append({
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "kind": kind,
                "height": height,
                "radius": round(radius, 3),
                "cells": cells,
            })

    if not regions:
        return []

    # Score within a kind, so a room full of unobserved floor cannot bury the
    # one cloud_only region that is a table in the robot's path.  Both are
    # reported; the robot does different things with them.
    largest: Dict[str, int] = {}
    for r in regions:
        largest[r["kind"]] = max(largest.get(r["kind"], 0), r["cells"])
    for r in regions:
        r["score"] = round(r["cells"] / largest[r["kind"]], 4)

    regions.sort(key=lambda r: (r["score"], r["cells"]), reverse=True)
    return regions[:max_regions]


def agreement_payload(
    stats: Dict[str, Any],
    regions: List[Dict[str, Any]],
    cover: Dict[str, float],
    *,
    cloud_points: int = 0,
) -> Dict[str, Any]:
    """Assemble the verdict the robot's ``MapCloudAgreement`` is built from.

    One flat dict, with the region list already split into the parallel arrays
    the ROS message uses.  Splitting it here rather than on the robot keeps the
    ordering guarantee in one place: the four arrays are parallel, and a bridge
    that zipped them back together from a list of dicts would be a second place
    for that to go wrong.

    ``nan`` heights are sent as ``null``.  JSON has no NaN, and a bridge reading
    a literal ``NaN`` token out of a non-strict parser and publishing it into a
    ``float32[]`` would give the robot a height that fails every comparison
    silently -- including ``height > robot_height``, which is the test that
    decides whether it drives into the thing.
    """
    return {
        "grid_coverage": cover.get("grid_coverage", 0.0),
        "cloud_coverage": cover.get("cloud_coverage", 0.0),
        "agreement": stats.get("agreement", 0.0),
        "compared_cells": int(stats.get("cloud_cells", 0)) + int(stats.get("missing", 0)),
        "conflicting_cells": int(stats.get("new_cells", 0)) + int(stats.get("missing", 0)),
        "cloud_points": int(cloud_points),
        "regions": len(regions),
        "uncertain_x": [r["x"] for r in regions],
        "uncertain_y": [r["y"] for r in regions],
        "uncertain_scores": [r["score"] for r in regions],
        "uncertain_kinds": [r["kind"] for r in regions],
        "uncertain_heights": [
            None if not np.isfinite(r["height"]) else round(float(r["height"]), 3)
            for r in regions
        ],
        "uncertain_radii": [r["radius"] for r in regions],
    }
