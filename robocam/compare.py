"""Compare: the reconstruction against the robot's own 2D map.

The box in the middle of the server's column.  It takes the point cloud the
reconstruction produced, puts it in the frame the robot's occupancy grid is drawn
in, and asks what the two disagree about.  What comes out is the solid arrow back
to the robot's map (:func:`compare`, returning a patch) and the dotted one to
SLAM (:func:`estimate_shift`, returning an offer of a correction).

What this is actually for
-------------------------
The robot's grid comes from an LDS-02: one horizontal plane, at the height the
scanner is bolted at.  It cannot see a table top, a shelf edge, a windowsill, an
open drawer, or the tray of a trolley — and a robot with a camera mast or a
gripper collides with all of them.  A monocular reconstruction sees the whole
volume in front of the camera.  So the disagreement worth reporting is
overwhelmingly one-directional: *the cloud has surfaces in cells the scan says
are free*, at heights the scan never visited.

That asymmetry is why the output is only ever additive.  See
:func:`robocam.occupancy.apply_patch` for the merge rule; the short version is
that a reconstruction which invents a surface costs a detour, while one that
erases a real wall costs a collision, so this stage is permitted to do the first
and never the second.

Putting the cloud in the map frame
----------------------------------
The cloud does not arrive in the map frame and cannot: CUT3R's world frame is
anchored on the first frame of the current ``map_id``, with an origin and an
orientation nobody chose.  What relates them is the camera pose, known in both:

    T_map_cut3r  =  T_map_base · T_base_cam · (T_cut3r_cam)⁻¹

``T_map_base`` is the odometry pose that arrived with the frame, ``T_base_cam``
is where the camera is bolted (config, not guessable), and ``T_cut3r_cam`` is the
pose the model returned with the cloud.  Get any of the three wrong and the cloud
lands somewhere plausible and false, which is why :func:`compare` reports
``agreement`` — the share of cloud surface that coincides with mapped obstacle —
as its first-class health number.  A correct placement in a mapped room agrees
substantially; a broken one agrees at chance.

Two conventions meet here and both are needed
---------------------------------------------
The model works in the vision convention (x right, y **down**, z forward out of
the lens).  The robot works in REP-103 (x forward, y left, z **up**).  The
rotation between them is fixed and is applied here rather than being folded into
the configured mount angles, because a configuration value that silently carries
a 90° convention flip inside it is a value nobody can check against a tape
measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .occupancy import Grid, patch_from_diff
from .odometry import Odom

#: Vision-optical (x right, y down, z forward) -> body (x forward, y left, z up).
#: Fixed, exact, and written out rather than composed from Euler angles so that
#: it can be read off and checked: the optical z axis becomes body x, optical x
#: becomes body -y, optical y becomes body -z.
OPTICAL_TO_BODY = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


@dataclass
class CameraMount:
    """Where the camera sits on the robot, in the base frame.

    Metres and radians, REP-103.  These are tape-measure facts about this robot,
    and the reason they are configuration rather than constants is that the
    Mecanumbot's camera is on a neck motor — ``pitch`` in particular is whatever
    the neck is currently commanded to, and a stage that wants a moving neck can
    update it per frame instead of per session.
    """

    x: float = 0.10
    y: float = 0.0
    z: float = 0.45
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def matrix(self) -> np.ndarray:
        """The 4x4 base <- optical transform, convention flip included."""
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        # ZYX (yaw, pitch, roll), the same order REP-103 and every ROS tool use.
        rot = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)
        out = np.eye(4)
        out[:3, :3] = rot @ OPTICAL_TO_BODY
        out[:3, 3] = (self.x, self.y, self.z)
        return out


@dataclass
class CompareResult:
    """What one comparison found."""

    #: The cells to send back, or None when nothing differed.
    patch: Optional[Grid] = None
    #: Numbers for the reply and the log.  See :func:`compare` for what they mean.
    stats: Dict[str, Any] = field(default_factory=dict)
    #: The full updated grid, for a caller that wants to keep a merged copy.
    updated: Optional[np.ndarray] = None
    #: The boolean "the cloud says occupied" grid this comparison was made from,
    #: and the cloud in map-frame metres.  Kept so that :mod:`robocam.regions`
    #: can cluster the disagreement without redoing the transform and the
    #: binning -- both of which cost more than the clustering does, and neither
    #: of which would be guaranteed to reproduce the same cells if a parameter
    #: drifted between the two calls.  ``None`` when the comparison refused.
    cloud_occupied: Optional[np.ndarray] = None
    points_map: Optional[np.ndarray] = None

    @property
    def ok(self) -> bool:
        return bool(self.stats.get("placed"))


def invert_rigid(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform, by transpose and negate.

    Not ``np.linalg.inv``: for a rotation the transpose *is* the inverse and is
    exact, while a general inversion of a nearly-singular matrix is neither.
    """
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rot, t = matrix[:3, :3], matrix[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ t
    return out


def map_from_cloud_matrix(pose_c2w: np.ndarray, odom: Odom,
                          mount: CameraMount) -> np.ndarray:
    """``T_map_cloud``: the 4x4 that takes the model's world frame to the map frame.

    The composition from the module docstring, in one place so that the two
    directions cannot drift apart.  Both :func:`cloud_to_map` and
    :func:`map_to_cloud` are this matrix and its inverse, which is what makes
    "the coordinate I sent the robot, taken back to the cloud" round-trip
    exactly rather than approximately.
    """
    pose_c2w = np.asarray(pose_c2w, dtype=np.float64).reshape(4, 4)

    # T_map_base: the robot's planar pose, lifted to 3D.  The robot drives on a
    # floor, so roll and pitch of the base are zero by construction here; if that
    # ever stops being true the odometry message carries a quaternion and this is
    # where it would be used.
    c, s = math.cos(odom.yaw), math.sin(odom.yaw)
    t_map_base = np.eye(4)
    t_map_base[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t_map_base[:3, 3] = (odom.x, odom.y, odom.z)

    t_map_cam = t_map_base @ mount.matrix()
    return t_map_cam @ invert_rigid(pose_c2w)


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return points
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def cloud_to_map(points: np.ndarray, pose_c2w: np.ndarray, odom: Odom,
                 mount: CameraMount) -> np.ndarray:
    """Transform a cloud from the model's world frame into the robot's map frame.

    ``points`` is (N, 3) in the reconstruction's world frame; ``pose_c2w`` is the
    4x4 camera-to-that-world pose the model returned for the same frame.  This is
    the direction everything in T1 uses: the reconstruction is produced in the
    model's frame and has to be compared against a grid drawn in the robot's.
    """
    return _apply(map_from_cloud_matrix(pose_c2w, odom, mount), points)


def map_to_cloud(points_map: np.ndarray, pose_c2w: np.ndarray, odom: Odom,
                 mount: CameraMount) -> np.ndarray:
    """The other direction: a map-frame coordinate back into the model's world frame.

    T2 needs this and T1 does not, which is why it arrived second.  Once the
    robot has been told "the mug is at (3.2, 1.4) in the map", every subsequent
    question about that place — is there still something there, what does the
    current cloud say about its height — is a question about a region of the
    *cloud*, and the coordinate has to travel back across the same transform it
    came out of.

    The caveat that matters: ``pose_c2w`` anchors this to one reconstruction
    state.  CUT3R's world frame is reset whenever ``map_id`` changes, so a
    coordinate carried back with a pose from a different ``map_id`` lands
    somewhere arbitrary.  Callers hold the ``map_id`` alongside the matrix for
    exactly this reason; see :class:`robocam.seek.Placement`.
    """
    return _apply(invert_rigid(map_from_cloud_matrix(pose_c2w, odom, mount)), points_map)


def occupancy_from_cloud(
    points_map: np.ndarray,
    grid: Grid,
    *,
    z_min: float = 0.05,
    z_max: float = 1.6,
    min_points: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Bin a map-frame cloud into a boolean "the cloud says occupied" grid.

    The height slice is the whole point.  ``z_min`` excludes the floor, which is
    the largest surface any indoor reconstruction produces and which is not an
    obstacle; ``z_max`` excludes the ceiling and anything the robot passes under.
    Both are heights **above the map frame's floor plane**, so they are only
    meaningful once the placement is right — which is the other reason
    :func:`compare` reports agreement.

    ``min_points`` is what separates a surface from a speck.  A single confident
    point in a cell is one pixel of a monocular depth estimate and is not
    evidence of anything; three in a 5 cm cell is a patch of something real.
    """
    stats: Dict[str, Any] = {"points": int(points_map.shape[0])}
    occupied = np.zeros(grid.cells.shape, dtype=bool)
    if points_map.size == 0:
        stats.update({"in_slice": 0, "in_bounds": 0, "cells": 0})
        return occupied, stats

    z = points_map[:, 2]
    in_slice = (z >= z_min) & (z <= z_max)
    stats["in_slice"] = int(in_slice.sum())
    stats["below_slice"] = int((z < z_min).sum())
    stats["above_slice"] = int((z > z_max).sum())
    if not in_slice.any():
        stats.update({"in_bounds": 0, "cells": 0})
        return occupied, stats

    sliced = points_map[in_slice]
    col, row = grid.world_to_cell(sliced[:, 0], sliced[:, 1])
    keep = grid.in_bounds(col, row)
    stats["in_bounds"] = int(keep.sum())
    if not keep.any():
        stats["cells"] = 0
        return occupied, stats

    # One bincount over flattened indices rather than a loop or a dict: this runs
    # per frame on the worker thread and a 100k-point cloud must not cost more
    # than the model already did.
    flat = row[keep] * grid.width + col[keep]
    hits = np.bincount(flat, minlength=grid.size)
    occupied = (hits >= min_points).reshape(grid.cells.shape)
    stats["cells"] = int(occupied.sum())
    stats["max_points_per_cell"] = int(hits.max()) if hits.size else 0
    return occupied, stats


def estimate_shift(
    cloud_occupied: np.ndarray,
    grid: Grid,
    *,
    search_cells: int = 4,
    min_inliers: int = 40,
    min_gain: float = 1.15,
) -> Optional[Dict[str, Any]]:
    """Look for a translation that would make the cloud agree with the map better.

    A brute-force search over integer cell offsets, scoring each by how many
    cloud-occupied cells land on map-occupied cells.  Returns None when the best
    offset is the zero one, when too little overlaps to mean anything, or when
    the best offset is not clearly better than staying put.

    What this can and cannot do, stated plainly because a pose correction that
    is quietly wrong is worse than none:

    * It searches **translation only**.  A yaw error does not present as a
      translation, so it shows up here as a search that finds no offset worth
      reporting — which is the safe failure, not a wrong hint.
    * Its resolution is one cell, so it cannot offer a correction finer than the
      map itself.  For a 5 cm grid that is the right granularity for the drift
      this is meant to catch (tens of centimetres over a run), and far too coarse
      to be used as a localisation source, which it is not.
    * ``min_gain`` guards against the case that produces the most convincing
      nonsense: a corridor, where sliding the cloud a few cells along the wall
      scores almost identically everywhere and the argmax is noise.

    The result is an *offer*.  ``pose_hint`` says ``advisory: true`` and means it;
    SLAM has a pose graph and this has a histogram.
    """
    occupied_map, _, _ = grid.classify()
    if not cloud_occupied.any() or not occupied_map.any():
        return None

    def score_at(dy: int, dx: int) -> int:
        # Roll rather than pad-and-slice: the wrap-around only affects the
        # outermost `search_cells` rows/columns, the search is a handful of
        # cells wide, and a map's border is unknown space that scores zero.
        shifted = np.roll(np.roll(cloud_occupied, dy, axis=0), dx, axis=1)
        return int(np.count_nonzero(shifted & occupied_map))

    base = score_at(0, 0)
    best = (base, 0, 0)
    for dy in range(-search_cells, search_cells + 1):
        for dx in range(-search_cells, search_cells + 1):
            if dx == 0 and dy == 0:
                continue
            s = score_at(dy, dx)
            if s > best[0]:
                best = (s, dy, dx)

    score, dy, dx = best
    if (dx, dy) == (0, 0) or score < min_inliers:
        return None
    if base > 0 and score < base * min_gain:
        return None

    return {
        # The correction is what the *robot's pose* would have to move by for the
        # cloud to land where the map says it should — the same direction the
        # cloud was shifted, since the cloud is rigidly attached to the pose.
        "dx": dx * grid.resolution,
        "dy": dy * grid.resolution,
        "dyaw": 0.0,
        "inliers": score,
        "baseline_inliers": base,
        "confidence": round(min(1.0, score / max(1, int(cloud_occupied.sum()))), 3),
        "method": "grid_shift",
        "search_cells": search_cells,
    }


def compare(
    points: np.ndarray,
    pose_c2w: np.ndarray,
    odom: Odom,
    grid: Grid,
    mount: CameraMount,
    *,
    z_min: float = 0.05,
    z_max: float = 1.6,
    min_points: int = 3,
    occupied_value: int = 100,
    min_new_cells: int = 4,
    hint: bool = True,
    search_cells: int = 4,
) -> CompareResult:
    """One frame's comparison: cloud in, patch and numbers out.

    The reported statistics, and what each is for:

    ``agreement``   share of cloud-occupied cells that the map also calls
                    occupied.  **The health number.**  A correct placement in a
                    mapped room is well above chance; a broken transform, a wrong
                    camera mount or an odometry pose in the wrong frame all drive
                    this towards zero, and none of them announce themselves any
                    other way.
    ``new_cells``   cells the cloud calls occupied that the map calls free or
                    unknown.  This is the product — the table tops.
    ``over_free``   the subset of those that the map explicitly calls *free*,
                    i.e. cells the scanner looked at and found empty.  Worth
                    separating: a surface over unknown space is merely unmapped,
                    while a surface over swept-free space is exactly the
                    out-of-plane obstacle this whole stage exists to find.
    ``missing``     cells the map calls occupied and the cloud does not.  Never
                    acted on — see the module docstring — but a run where this
                    dwarfs everything else means the cloud is not where it should
                    be, and it is the second place after ``agreement`` to look.

    ``min_new_cells`` suppresses the patch when the news amounts to a few
    scattered cells.  Those are single-frame reconstruction noise far more often
    than they are furniture, and a robot that receives an obstacle for every one
    of them ends up in a costmap full of specks.
    """
    result = CompareResult()

    if grid.frame != odom.frame:
        # The one refusal in this module.  A pose in `odom` and a grid in `map`
        # differ by SLAM's accumulated correction, so comparing across them
        # produces a confident, wrong answer rather than an obviously broken one.
        result.stats = {
            "placed": False,
            "error": (f"pose is in frame {odom.frame!r} but the grid is in "
                      f"{grid.frame!r}; refusing to compare across frames"),
        }
        return result

    points_map = cloud_to_map(points, pose_c2w, odom, mount)
    cloud_occupied, bin_stats = occupancy_from_cloud(
        points_map, grid, z_min=z_min, z_max=z_max, min_points=min_points,
    )
    result.cloud_occupied = cloud_occupied
    result.points_map = points_map

    occupied_map, free_map, unknown_map = grid.classify()
    agree = cloud_occupied & occupied_map
    new = cloud_occupied & ~occupied_map
    over_free = new & free_map
    missing = occupied_map & ~cloud_occupied

    cloud_cells = int(cloud_occupied.sum())
    stats: Dict[str, Any] = {
        "placed": True,
        "frame": grid.frame,
        "map_id": grid.map_id,
        "cloud_cells": cloud_cells,
        "agree_cells": int(agree.sum()),
        "agreement": round(int(agree.sum()) / cloud_cells, 4) if cloud_cells else 0.0,
        "new_cells": int(new.sum()),
        "over_free": int(over_free.sum()),
        "over_unknown": int((new & unknown_map).sum()),
        "missing": int(missing.sum()),
        "z_slice": [z_min, z_max],
        **bin_stats,
    }

    if int(new.sum()) < min_new_cells:
        stats["patched"] = False
        result.stats = stats
        return result

    updated = grid.cells.copy()
    updated[new] = np.int8(occupied_value)
    patch = patch_from_diff(grid, updated)
    stats["patched"] = patch is not None
    if patch is not None:
        stats["patch_cells"] = patch.size
        stats["patch_rect"] = [patch.x0, patch.y0, patch.width, patch.height]

    result.patch = patch
    result.updated = updated
    result.stats = stats

    if hint:
        shift = estimate_shift(cloud_occupied, grid, search_cells=search_cells)
        if shift is not None:
            stats["shift"] = shift
    return result
