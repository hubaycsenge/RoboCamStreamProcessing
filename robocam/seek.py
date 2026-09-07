"""The decision stage: from a box in the image to a goal in the robot's map.

The ``LLM/decision making`` box of the system diagram, minus the detector, which
is :mod:`robocam.detect`.  Everything here is geometry and policy: no weights, no
CUDA, no network, and therefore testable.  That split is the point — the half of
this stage that can be wrong in a way nobody notices is the geometry, and
geometry that needs a GPU to exercise does not get exercised.

What it produces
----------------
One :class:`Sighting` — the target's position in the **map frame**, with the
evidence that put it there — and one :class:`Reach` verdict saying whether the
robot can actually pick the thing up.  Together those are the ``found``
announcement, which is the only message in this system that ends a mission.

Why the reachability verdict is here and not on the robot
---------------------------------------------------------
The robot's LiDAR sees one horizontal plane.  Asked "is the mug on the floor or
on the table", it has nothing to say — the two are the same reading.  The height
of a surface is precisely what the reconstruction knows and the scanner does not,
so the server is the only end of this link that can answer, and answering it
badly costs the robot a minute of driving to something it was never going to be
able to lift.  So ``found`` carries the verdict rather than the raw z, and the
behaviour tree can branch on it without knowing what a gripper envelope is.

Two ways to know where the target is
------------------------------------
``basis: "live"``    the detector is looking at it right now, in this frame.
``basis: "memory"``  it was seen earlier — in T1, before the target was even
                     named — and the coordinate comes from a stored keyframe.

The second is what makes the mission's shape work.  T1 explores without knowing
what it will be asked for; the target is named at T2 launch.  So T1 keeps
keyframes (see :class:`KeyframeStore`) — the image, and the transform that puts
that image's cloud in the map frame — and at T2 launch the newly-named target is
searched for in what T1 already saw.  A hit gives the robot a coordinate to drive
to before it has taken a single frame of T2, which is exactly the "go to where
you last saw it" branch of the behaviour tree.

A ``found`` from memory is not a claim that the object is there now.  It is a
claim about where it was and when, and ``age_s`` says how stale that is.  The
robot's own second branch — watching for the object the whole way — is what
resolves the difference, and the object having moved is the case the search
behaviour exists for.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .compare import CameraMount, invert_rigid, map_from_cloud_matrix
from .detect import Detection
from .occupancy import Grid
from .odometry import Odom, wrap_angle

log = logging.getLogger(__name__)

#: Where a coordinate came from.  On the wire in ``found.basis``.
BASIS_LIVE = "live"
BASIS_MEMORY = "memory"

#: Reachability verdicts.  ``unknown`` is a real answer and the honest one when
#: the height could not be measured; the robot should approach and look again
#: rather than treat it as either yes or no.
REACH_OK = "reachable"
REACH_TOO_HIGH = "too_high"
REACH_BELOW_FLOOR = "below_floor"
REACH_NO_STANDING_ROOM = "no_standing_room"
REACH_UNKNOWN = "unknown"


# -- placing a cloud ---------------------------------------------------------

@dataclass
class Placement:
    """A cloud, and everything needed to move coordinates between it and the map.

    Holds the three transforms of the composition in :mod:`robocam.compare` at
    the instant one frame was reconstructed, plus the two identities that say
    which cloud and which map they belong to.  Those identities are not
    decoration: CUT3R restarts its world frame on every reset, so a matrix kept
    past a ``cloud_map_id`` change converts into a frame that no longer exists,
    and a coordinate kept past a ``map_id`` change names a place in a map the
    robot has thrown away.  Both are checked before use rather than trusted.
    """

    pose_c2w: np.ndarray
    odom: Odom
    mount: CameraMount
    #: The robot's map identity — which SLAM map these coordinates are in.
    map_id: str = ""
    #: The reconstruction's own identity, incremented on every CUT3R reset.
    cloud_map_id: int = 0
    frame: str = "map"
    t_ns: int = 0

    def matrix(self) -> np.ndarray:
        """``T_map_cloud``."""
        return map_from_cloud_matrix(self.pose_c2w, self.odom, self.mount)

    def to_map(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return pts
        m = self.matrix()
        return pts @ m[:3, :3].T + m[:3, 3]

    def to_cloud(self, points_map: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_map, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return pts
        m = invert_rigid(self.matrix())
        return pts @ m[:3, :3].T + m[:3, 3]

    @property
    def camera_centre(self) -> np.ndarray:
        """The camera's position in the cloud's own world frame."""
        return np.asarray(self.pose_c2w, dtype=np.float64).reshape(4, 4)[:3, 3]


@dataclass
class ViewGeometry:
    """How the model's pointmap pixels relate to the frame the detector saw.

    The reconstruction does not run on the frame that arrived: CUT3R resizes it
    and centre-crops to a multiple of 16, so pointmap pixel (0, 0) is not image
    pixel (0, 0) and the two grids have different sizes.  A detector box is in
    *image* coordinates and the points behind it are indexed in *pointmap*
    coordinates, and getting that mapping wrong does not raise — it silently
    reads the depth of whatever is a few centimetres to the left, which is a
    coordinate that looks entirely plausible and is not the object.

    So the processor records what it actually did, here, rather than anyone
    recomputing it from the config.
    """

    #: Scale applied to the source image before cropping.  Two numbers rather
    #: than one because the resize rounds each axis to whole pixels
    #: independently, so a 1280x720 frame does not scale by exactly the same
    #: factor in x and y — a fraction of a percent, which is a pixel or two at
    #: the edge of a 512-wide crop and worth not throwing away.
    scale_x: float = 1.0
    scale_y: float = 1.0
    #: Top-left of the crop, in the *resized* image's pixels.
    crop_x0: int = 0
    crop_y0: int = 0
    #: The pointmap's own size, which is the crop's size.
    out_w: int = 0
    out_h: int = 0
    #: The source frame's size, kept so a box can be validated against it.
    src_w: int = 0
    src_h: int = 0

    def image_to_pointmap(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale_x - self.crop_x0, y * self.scale_y - self.crop_y0)

    def pointmap_to_image(self, u: float, v: float) -> Tuple[float, float]:
        return ((u + self.crop_x0) / self.scale_x, (v + self.crop_y0) / self.scale_y)

    def box_to_pointmap(self, box: Sequence[float]) -> Optional[Tuple[int, int, int, int]]:
        """Map an image box into pointmap indices, or None if nothing overlaps.

        None rather than an empty box, because "the object is outside the
        reconstructed crop" is a distinct and common situation — the crop is
        narrower than the camera's frame — and it is not the same as "the object
        is not there".  The caller reports it as such.
        """
        x0, y0 = self.image_to_pointmap(box[0], box[1])
        x1, y1 = self.image_to_pointmap(box[2], box[3])
        u0 = int(max(0, math.floor(min(x0, x1))))
        v0 = int(max(0, math.floor(min(y0, y1))))
        u1 = int(min(self.out_w, math.ceil(max(x0, x1))))
        v1 = int(min(self.out_h, math.ceil(max(y0, y1))))
        if u1 <= u0 or v1 <= v0:
            return None
        return (u0, v0, u1, v1)


def shrink_box(box: Sequence[int], keep: float = 0.7) -> Tuple[int, int, int, int]:
    """Pull a box in towards its centre, keeping ``keep`` of each side.

    A detector box is a bounding box, so its corners are background — the wall
    behind the mug, the desk beside it — and the corners are also where a
    monocular reconstruction is least reliable, because a depth discontinuity
    runs right through them.  Sampling the middle costs a few pixels of the
    object and removes most of the surface that is not it.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) * keep / 2.0, (y1 - y0) * keep / 2.0
    return (int(math.floor(cx - hw)), int(math.floor(cy - hh)),
            int(math.ceil(cx + hw)), int(math.ceil(cy + hh)))


# -- what a sighting is ------------------------------------------------------

@dataclass
class Sighting:
    """The target, located, in the robot's map frame."""

    target: str
    x: float
    y: float
    z: float
    confidence: float
    frame: str = "map"
    map_id: str = ""
    basis: str = BASIS_LIVE
    #: Server monotonic ns when the frame this came from was reconstructed.
    t_ns: int = 0
    #: The detector's box in the source frame, for the log and the overlay.
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    #: How many cloud points survived to produce the centroid.  The single best
    #: indicator of whether the coordinate is worth anything: a box backed by six
    #: points is a guess, one backed by four hundred is a measurement.
    points: int = 0
    #: Metres from the camera to the target, from the cloud rather than from any
    #: assumption about object size.
    range_m: float = 0.0
    #: Spread of the surviving points, metres.  A mug is a few centimetres; half
    #: a metre means the box caught the object and the wall behind it.
    spread_m: float = 0.0
    #: Bearing from the robot's base, radians, positive left.
    bearing_rad: float = 0.0
    detector: str = ""
    query: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def age_s(self, now_ns: Optional[int] = None) -> float:
        now = time.monotonic_ns() if now_ns is None else now_ns
        return max(0.0, (now - self.t_ns) / 1e9) if self.t_ns else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "x": round(self.x, 4), "y": round(self.y, 4), "z": round(self.z, 4),
            "confidence": round(self.confidence, 4),
            "frame": self.frame, "map_id": self.map_id, "basis": self.basis,
            "box": [int(v) for v in self.box],
            "points": self.points,
            "range_m": round(self.range_m, 3),
            "spread_m": round(self.spread_m, 3),
            "bearing_deg": round(math.degrees(self.bearing_rad), 1),
            "detector": self.detector,
            "query": self.query,
            **({"extra": self.extra} if self.extra else {}),
        }


def locate(
    detection: Detection,
    pointmap: np.ndarray,
    conf: np.ndarray,
    view: ViewGeometry,
    placement: Placement,
    target: str,
    *,
    min_conf: float = 1.5,
    min_points: int = 8,
    depth_band_m: float = 0.25,
    keep_frac: float = 0.7,
    detector: str = "",
) -> Tuple[Optional[Sighting], Dict[str, Any]]:
    """Put one detection in the map frame.  Returns (sighting or None, why).

    The three filters, in the order they matter:

    **The crop.**  The box is in the frame's coordinates and the pointmap covers
    only the centre crop the model ran on, so a target at the edge of the camera's
    view has a box and no points.  Reported as ``outside_reconstruction`` rather
    than as absence — the robot should turn towards it, not conclude it is gone.

    **Confidence.**  The same ``conf_self`` gate the cloud itself uses.  A box
    over a window or a specular surface is where a monocular model is least sure
    and most wrong, and its unconfident points would drag the centroid metres.

    **Depth.**  This is the one that decides whether the coordinate is usable.  A
    bounding box always contains background, so the naive centroid of everything
    behind it sits between the object and the wall — for a mug on a desk against
    a far wall, typically a metre past the mug.  Taking the *median* range first
    and keeping only points within ``depth_band_m`` of it picks the dominant
    surface in the box, which for a box that is mostly object is the object.

    The last filter is also the honest failure mode: when the box is mostly wall,
    the dominant surface is the wall and the answer is confidently wrong.
    ``spread_m`` and ``points`` in the sighting are what a reader uses to catch
    that, which is why they travel all the way to the robot rather than staying
    in a log line.
    """
    why: Dict[str, Any] = {"box": [int(v) for v in detection.box]}

    sub = view.box_to_pointmap(shrink_box(detection.box, keep_frac))
    if sub is None:
        why["reason"] = "outside_reconstruction"
        why["detail"] = ("the detection is outside the centre crop the model "
                         "reconstructs; turn towards it to bring it inside")
        return None, why
    u0, v0, u1, v1 = sub
    why["pointmap_box"] = [u0, v0, u1, v1]

    pm = np.asarray(pointmap, dtype=np.float64)
    if pm.ndim != 3 or pm.shape[2] != 3:
        pm = pm.reshape(view.out_h, view.out_w, 3)
    cf = np.asarray(conf, dtype=np.float64).reshape(view.out_h, view.out_w)

    patch = pm[v0:v1, u0:u1].reshape(-1, 3)
    patch_conf = cf[v0:v1, u0:u1].reshape(-1)
    why["pixels"] = int(patch.shape[0])

    good = np.isfinite(patch).all(axis=1) & (patch_conf >= min_conf)
    why["confident"] = int(good.sum())
    if int(good.sum()) < min_points:
        why["reason"] = "too_few_confident_points"
        why["detail"] = (f"{int(good.sum())} points over conf {min_conf} in the box, "
                         f"{min_points} needed; the model is unsure about this surface")
        return None, why
    patch = patch[good]

    ranges = np.linalg.norm(patch - placement.camera_centre, axis=1)
    median = float(np.median(ranges))
    near = np.abs(ranges - median) <= depth_band_m
    why["in_depth_band"] = int(near.sum())
    if int(near.sum()) < min_points:
        why["reason"] = "no_dominant_surface"
        why["detail"] = ("the points behind the box are spread over depth with no "
                         "dominant surface; the box is probably mostly background")
        return None, why
    kept = patch[near]

    centroid_cloud = kept.mean(axis=0)
    centroid_map = placement.to_map(centroid_cloud[None, :])[0]
    spread = float(np.linalg.norm(kept.std(axis=0)))

    dx = float(centroid_map[0]) - placement.odom.x
    dy = float(centroid_map[1]) - placement.odom.y
    bearing = wrap_angle(math.atan2(dy, dx) - placement.odom.yaw)

    sighting = Sighting(
        target=target,
        x=float(centroid_map[0]), y=float(centroid_map[1]), z=float(centroid_map[2]),
        confidence=float(detection.score),
        frame=placement.frame,
        map_id=placement.map_id,
        basis=BASIS_LIVE,
        t_ns=placement.t_ns or time.monotonic_ns(),
        box=tuple(int(v) for v in detection.box),
        points=int(kept.shape[0]),
        range_m=median,
        spread_m=spread,
        bearing_rad=bearing,
        detector=detector,
        query=detection.query or detection.label,
    )
    why["reason"] = "located"
    return sighting, why


# -- can the robot actually pick it up? --------------------------------------

@dataclass
class ReachEnvelope:
    """What this robot can reach, in metres.  Tape-measure facts, like the mount.

    ``grasp_z_max`` is the one that does the work and the one worth measuring.
    The Mecanumbot's gripper closes at floor level, so an object more than a hand's
    height above the floor is not a thing it can pick up however close it drives —
    and telling the robot that *before* it drives is the whole value of having a
    reconstruction rather than a scanner.
    """

    #: Below this the point is under the floor, which is a reconstruction error
    #: rather than an object.  Slightly negative because the map's floor plane is
    #: a plane and a real floor is not.
    grasp_z_min: float = -0.05
    #: Top of the gripper's vertical envelope, above the floor.
    grasp_z_max: float = 0.12
    #: How far from the base centre the gripper closes.  Used to ask whether any
    #: standing position exists from which the object is in range.
    reach_radius_m: float = 0.45
    #: How far short of the object the robot should stop, for the nav goal.
    standoff_m: float = 0.8
    #: Footprint radius, for deciding whether a candidate standing cell is clear.
    robot_radius_m: float = 0.22


@dataclass
class Reach:
    """Whether the robot can act on the sighting, and why not when it cannot."""

    reachable: Optional[bool]
    verdict: str
    z_above_floor: float = 0.0
    #: Where the robot should stand, in the map frame: (x, y, yaw).
    approach: Optional[Tuple[float, float, float]] = None
    #: Free-text, for the log.  Nothing on the robot parses it.
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "reachable": self.reachable,
            "verdict": self.verdict,
            "z_above_floor": round(self.z_above_floor, 3),
            "reason": self.reason,
        }
        out.update(self.detail)
        return out


def _cell_is_clear(grid: Grid, x: float, y: float, radius_m: float) -> Optional[bool]:
    """Is a disc of ``radius_m`` at (x, y) free of mapped obstacle?

    None when the disc falls outside the grid, which is a third answer and not a
    synonym for either: off the edge of the map is unknown space the robot may
    well be able to stand in, and treating it as blocked would refuse every
    approach at the frontier — exactly where seeking happens.
    """
    cells = max(1, int(round(radius_m / max(grid.resolution, 1e-6))))
    col, row = grid.world_to_cell(np.array([x]), np.array([y]))
    c, r = int(col[0]), int(row[0])
    if not (0 <= c < grid.width and 0 <= r < grid.height):
        return None
    c0, c1 = max(0, c - cells), min(grid.width, c + cells + 1)
    r0, r1 = max(0, r - cells), min(grid.height, r + cells + 1)
    window = grid.cells[r0:r1, c0:c1]
    return not bool(np.any(window >= grid.occupied_min))


def choose_approach(
    sighting: Sighting,
    envelope: ReachEnvelope,
    grid: Optional[Grid] = None,
    from_xy: Optional[Tuple[float, float]] = None,
    *,
    candidates: int = 12,
) -> Tuple[Optional[Tuple[float, float, float]], str, Dict[str, Any]]:
    """Where to stand.  Returns (pose or None, note, detail).

    First choice is the standoff pose on the line the robot is already on — it is
    the shortest drive and the one whose heading the robot is closest to already.
    When that lands on an obstacle (a mug on the far side of a table puts the
    natural approach inside the table), a ring of alternatives at the same radius
    is tried, nearest-in-angle first, so the fallback stays as close to the
    natural approach as the furniture allows.

    Returning None is a real outcome and the one the ``no_standing_room`` verdict
    is made of: an object with no free cell around it cannot be collected, and
    the robot is better off told that than sent to bump into the table.
    """
    detail: Dict[str, Any] = {}
    ox, oy = (from_xy if from_xy is not None else (0.0, 0.0))
    dx, dy = sighting.x - ox, sighting.y - oy
    base_heading = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else 0.0

    def pose_at(heading: float) -> Tuple[float, float, float]:
        # Stand `standoff` away from the object along `heading`, facing back
        # towards it.  The facing is heading + pi, not heading, and that sign is
        # the difference between arriving in front of the object and arriving
        # with it behind the robot.
        px = sighting.x - envelope.standoff_m * math.cos(heading)
        py = sighting.y - envelope.standoff_m * math.sin(heading)
        return (px, py, wrap_angle(math.atan2(sighting.y - py, sighting.x - px)))

    natural = pose_at(base_heading)
    if grid is None:
        detail["approach_checked"] = False
        return natural, "no grid to check the approach against", detail

    detail["approach_checked"] = True
    clear = _cell_is_clear(grid, natural[0], natural[1], envelope.robot_radius_m)
    if clear is not False:
        detail["approach_source"] = "direct" if clear else "direct, off-map"
        return natural, "", detail

    # The direct approach is blocked.  Walk outwards in angle from it.
    offsets = sorted(
        (2.0 * math.pi * i / candidates for i in range(1, candidates)),
        key=lambda a: abs(wrap_angle(a)),
    )
    for offset in offsets:
        cand = pose_at(wrap_angle(base_heading + offset))
        if _cell_is_clear(grid, cand[0], cand[1], envelope.robot_radius_m) is not False:
            detail["approach_source"] = "ring"
            detail["approach_offset_deg"] = round(math.degrees(wrap_angle(offset)), 1)
            return cand, "direct approach was blocked; went round", detail

    detail["approach_source"] = "none"
    return None, "no free standing position within the standoff radius", detail


def judge_reach(
    sighting: Sighting,
    envelope: ReachEnvelope,
    grid: Optional[Grid] = None,
    from_xy: Optional[Tuple[float, float]] = None,
    *,
    z_known: bool = True,
) -> Reach:
    """Can the robot pick this up, and if not, why not.

    The height test is the one that matters and it comes first: the reconstruction
    measures the object's height above the map's floor plane, and anything above
    the gripper's envelope is not collectable at any approach.  A robot told that
    can still be sent to look at the thing — knowing where the mug is has value
    even when it is on a shelf — but it will not waste a grasp attempt on it.

    ``z_known=False`` produces ``unknown`` rather than a guess.  That happens when
    the coordinate came from memory across a reconstruction reset, and inventing
    a height there would be worse than admitting to not having one: the robot's
    correct response to ``unknown`` is to drive over and look, which is exactly
    what it should do.
    """
    detail: Dict[str, Any] = {
        "grasp_z": [envelope.grasp_z_min, envelope.grasp_z_max],
        "reach_radius_m": envelope.reach_radius_m,
        "standoff_m": envelope.standoff_m,
    }
    approach, note, approach_detail = choose_approach(sighting, envelope, grid, from_xy)
    detail.update(approach_detail)
    if approach is not None:
        detail["approach_range_m"] = round(
            math.hypot(sighting.x - approach[0], sighting.y - approach[1]), 3)

    if not z_known:
        return Reach(reachable=None, verdict=REACH_UNKNOWN, z_above_floor=sighting.z,
                     approach=approach,
                     reason="no height for this coordinate; approach and look again",
                     detail=detail)

    if approach is None:
        return Reach(reachable=False, verdict=REACH_NO_STANDING_ROOM,
                     z_above_floor=sighting.z, approach=None,
                     reason=note or "nowhere to stand", detail=detail)

    if sighting.z < envelope.grasp_z_min:
        return Reach(
            reachable=False, verdict=REACH_BELOW_FLOOR, z_above_floor=sighting.z,
            approach=approach,
            reason=(f"{sighting.z:.2f} m is below the map's floor plane, which is a "
                    "reconstruction error rather than an object"),
            detail=detail)

    if sighting.z > envelope.grasp_z_max:
        return Reach(
            reachable=False, verdict=REACH_TOO_HIGH, z_above_floor=sighting.z,
            approach=approach,
            reason=(f"{sighting.z:.2f} m above the floor, over the gripper's "
                    f"{envelope.grasp_z_max:.2f} m envelope"),
            detail=detail)

    # Within the envelope vertically; the remaining question is whether any
    # standing position puts it in the arm's horizontal range.
    if detail.get("approach_range_m", 0.0) > envelope.reach_radius_m:
        detail["note"] = ("the standoff is further than the arm reaches; the robot "
                          "must close the last stretch before grasping")

    return Reach(reachable=True, verdict=REACH_OK, z_above_floor=sighting.z,
                 approach=approach,
                 reason=note or f"{sighting.z:.2f} m above the floor, within the gripper's envelope",
                 detail=detail)


# -- remembering ------------------------------------------------------------

@dataclass
class Keyframe:
    """One T1 frame kept so a target named later can be searched for in it.

    The image is kept at reduced size — the detector does not need 720p to find
    a mug that fills a tenth of the frame, and a session's worth of full frames
    is gigabytes.  What is *not* reduced is the placement: the transform is what
    makes a hit in an old image into a coordinate in the current map, and there
    is nothing to gain by approximating it.
    """

    image: np.ndarray
    view: ViewGeometry
    placement: Placement
    pointmap: np.ndarray
    conf: np.ndarray
    seq: int = 0
    t_ns: int = 0


class KeyframeStore:
    """A bounded, thinned memory of what T1 saw.

    Bounded because this runs for a whole exploration run and the server is a
    Slurm job with a memory limit; thinned by *distance travelled* rather than by
    time, because a robot parked for a minute produces sixty near-identical
    frames and none of them is new evidence.  What the search wants is coverage
    of the room, and coverage is a property of where the camera was.

    Eviction is oldest-first with one exception: it never evicts below
    ``min_keep`` frames, so a very short T1 still leaves something to search.
    """

    def __init__(self, max_frames: int = 240, min_spacing_m: float = 0.25,
                 min_spacing_rad: float = 0.35, min_keep: int = 8) -> None:
        self.max_frames = int(max_frames)
        self.min_spacing_m = float(min_spacing_m)
        self.min_spacing_rad = float(min_spacing_rad)
        self.min_keep = int(min_keep)
        self._frames: List[Keyframe] = []
        self._last_xy: Optional[Tuple[float, float]] = None
        self._last_yaw: Optional[float] = None
        self.considered = 0
        self.stored = 0
        self.evicted = 0

    def __len__(self) -> int:
        return len(self._frames)

    def should_keep(self, odom: Odom) -> bool:
        """Has the camera moved enough since the last keyframe to be worth one?"""
        if self._last_xy is None:
            return True
        moved = math.hypot(odom.x - self._last_xy[0], odom.y - self._last_xy[1])
        turned = abs(wrap_angle(odom.yaw - (self._last_yaw or 0.0)))
        return moved >= self.min_spacing_m or turned >= self.min_spacing_rad

    def add(self, keyframe: Keyframe) -> bool:
        self.considered += 1
        odom = keyframe.placement.odom
        if not self.should_keep(odom):
            return False
        self._frames.append(keyframe)
        self._last_xy = (odom.x, odom.y)
        self._last_yaw = odom.yaw
        self.stored += 1
        while len(self._frames) > self.max_frames:
            self._frames.pop(0)
            self.evicted += 1
        return True

    def newest_first(self) -> List[Keyframe]:
        """Search order.  Newest first: the most recent sighting of a thing that
        may have been passed several times is the one least likely to be stale."""
        return list(reversed(self._frames))

    def clear(self) -> None:
        self._frames.clear()
        self._last_xy = None
        self._last_yaw = None

    def stats(self) -> Dict[str, Any]:
        return {"held": len(self._frames), "considered": self.considered,
                "stored": self.stored, "evicted": self.evicted,
                "max_frames": self.max_frames}


class TargetMemory:
    """The best coordinate known for the target, and how old it is.

    Kept deliberately simple — one slot, best-confidence-wins within a recency
    window — because the alternative is a tracker, and a tracker is a thing that
    needs to be right about identity across frames.  This system does not need
    that: it needs one place to drive to, and the robot's own live branch
    supersedes the memory the moment it sees the object.

    A newer sighting always wins over an older one of similar confidence, because
    the object may have moved and the newest evidence is the least wrong.
    """

    def __init__(self, ttl_s: float = 900.0, prefer_newer_after_s: float = 20.0) -> None:
        self.ttl_s = float(ttl_s)
        self.prefer_newer_after_s = float(prefer_newer_after_s)
        self._best: Optional[Sighting] = None
        self.updates = 0

    def remember(self, sighting: Sighting) -> bool:
        """Store the sighting if it beats what is held.  Returns whether it did."""
        if sighting is None:
            return False
        held = self._best
        if held is None:
            self._best, self.updates = sighting, self.updates + 1
            return True
        if held.target != sighting.target:
            # A new mission: the old coordinate describes a different object.
            self._best, self.updates = sighting, self.updates + 1
            return True
        newer_by = (sighting.t_ns - held.t_ns) / 1e9
        if sighting.confidence >= held.confidence or newer_by >= self.prefer_newer_after_s:
            self._best, self.updates = sighting, self.updates + 1
            return True
        return False

    def recall(self, target: str = "", map_id: str = "",
               now_ns: Optional[int] = None) -> Optional[Sighting]:
        """The held sighting, if it is still about this target, map and era.

        The ``map_id`` check is the one that is easy to leave out and expensive to
        omit: after a SLAM reset the robot's map has new coordinates, and a
        remembered (x, y) from the old one names a place that no longer exists,
        while looking exactly like a valid goal.
        """
        held = self._best
        if held is None:
            return None
        if target and held.target != target:
            return None
        if map_id and held.map_id and held.map_id != map_id:
            return None
        if self.ttl_s and held.age_s(now_ns) > self.ttl_s:
            return None
        return held

    def forget(self) -> None:
        self._best = None


# -- when is T1 finished? ----------------------------------------------------

class CoverageTracker:
    """How fast the map is still growing.

    One number over a sliding window: observed cells per second.  It is what
    separates "the robot has not finished exploring" from "the robot is driving
    around a room it has already mapped", and neither the explored fraction nor
    the exit list can tell them apart on its own — a room with a closed door has
    an explored fraction well under one and no frontier left to visit.
    """

    def __init__(self, window_s: float = 30.0) -> None:
        self.window_s = float(window_s)
        self._samples: List[Tuple[float, int]] = []

    def observe(self, observed_cells: int, now: Optional[float] = None) -> None:
        t = time.monotonic() if now is None else now
        self._samples.append((t, int(observed_cells)))
        cutoff = t - self.window_s
        # Keep one sample older than the window so a rate can still be measured
        # at the moment the window first fills.
        while len(self._samples) > 2 and self._samples[1][0] < cutoff:
            self._samples.pop(0)

    def rate_cells_per_s(self) -> Optional[float]:
        if len(self._samples) < 2:
            return None
        (t0, c0), (t1, c1) = self._samples[0], self._samples[-1]
        dt = t1 - t0
        if dt < 1.0:
            return None
        return (c1 - c0) / dt

    def span_s(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]


@dataclass
class T1Criteria:
    """Thresholds for calling the first scan finished.  See :func:`t1_exit_criteria`."""

    #: Share of the grid that must have been observed at all.  Not 1.0 and never
    #: could be: a rectangular grid always contains cells behind walls.
    min_explored: float = 0.80
    #: Frontier candidates the robot still calls open and wide enough to enter.
    max_open_exits: int = 0
    #: New observed cells per second, below which the map has stopped growing.
    stall_cells_per_s: float = 15.0
    #: How long the growth rate must have been measurable before it counts.
    min_stall_window_s: float = 30.0
    #: Mean compare agreement.  A T1 whose reconstruction never agreed with the
    #: grid has produced a cloud nobody should search in T2.
    min_agreement: float = 0.15
    #: Floor on the run itself, so a first grid arriving before the robot has
    #: moved cannot satisfy everything at once.
    min_frames: int = 300
    min_runtime_s: float = 60.0


def t1_exit_criteria(
    grid: Optional[Grid],
    exits: Sequence[Dict[str, Any]],
    *,
    criteria: T1Criteria,
    coverage: Optional[CoverageTracker] = None,
    mean_agreement: Optional[float] = None,
    frames: int = 0,
    runtime_s: float = 0.0,
    min_exit_width_m: float = 0.6,
) -> Dict[str, Any]:
    """Is the first scan done?  Returns the verdict and every test behind it.

    The four questions, and why one alone is not enough:

    ``explored``    the share of the grid observed.  Necessary, and on its own
                    it never finishes: the grid is a rectangle and rooms are not,
                    so some fraction is permanently behind a wall.
    ``no_open_exits`` nothing left the robot itself calls worth driving to.  This
                    is the decisive one, and it is the robot's judgement rather
                    than the server's because the robot is what knows which
                    frontiers it has already tried — the same argument
                    :mod:`robocam.mission` makes for exits coming from the robot.
    ``not_growing`` the map has stopped gaining cells.  This is what catches the
                    honest end of exploration in a closed room, where the
                    explored fraction plateaus below any threshold and the exit
                    list has been empty for a while.
    ``placed``      the reconstruction agreed with the grid often enough to
                    believe it.  Not a coverage question at all, and the reason
                    it belongs here is that T1's *product* is the cloud, not the
                    grid: exiting T1 with a healthy map and a cloud that was
                    never correctly placed gives T2 nothing to find the target
                    in, and the failure appears an hour later as "the detector
                    never sees anything".

    ``ready`` is the conjunction, plus the floors on frames and runtime that stop
    a session satisfying all four before the robot has gone anywhere.  It is a
    *recommendation*: this function never changes the phase.  The transition is a
    decision about the mission and it stays with whoever is running it, which is
    also what makes T1 endable early by hand for a demo.
    """
    tests: Dict[str, Any] = {}
    reasons: List[str] = []

    explored = float(grid.summary.get("explored_fraction", 0.0)) if grid is not None else 0.0
    if grid is not None and not grid.summary:
        occupied, free, unknown = grid.classify()
        explored = float(grid.size - int(unknown.sum())) / grid.size if grid.size else 0.0
    tests["explored_fraction"] = round(explored, 4)
    ok_explored = grid is not None and explored >= criteria.min_explored
    tests["explored"] = ok_explored
    if not ok_explored:
        reasons.append(
            "no grid uploaded yet" if grid is None else
            f"{explored:.0%} of the grid observed, {criteria.min_explored:.0%} wanted")

    open_exits = [e for e in exits
                  if e.get("status") == "open"
                  and float(e.get("width_m", 0.0) or 0.0) >= min_exit_width_m]
    tests["open_exits"] = len(open_exits)
    ok_exits = len(open_exits) <= criteria.max_open_exits
    tests["no_open_exits"] = ok_exits
    if not ok_exits:
        reasons.append(f"{len(open_exits)} open exits still to visit")

    rate = coverage.rate_cells_per_s() if coverage is not None else None
    span = coverage.span_s() if coverage is not None else 0.0
    tests["growth_cells_per_s"] = None if rate is None else round(rate, 1)
    ok_growth = (rate is not None and span >= criteria.min_stall_window_s
                 and rate < criteria.stall_cells_per_s)
    tests["not_growing"] = ok_growth
    if not ok_growth:
        reasons.append(
            "not enough map history to say whether it is still growing"
            if rate is None or span < criteria.min_stall_window_s else
            f"map still growing at {rate:.0f} cells/s")

    tests["mean_agreement"] = None if mean_agreement is None else round(mean_agreement, 4)
    ok_placed = mean_agreement is not None and mean_agreement >= criteria.min_agreement
    tests["placed"] = ok_placed
    if not ok_placed:
        reasons.append(
            "the reconstruction has never been compared against the grid"
            if mean_agreement is None else
            f"cloud/grid agreement is {mean_agreement:.2f}, under {criteria.min_agreement:.2f} "
            "-- the reconstruction is not placed where the map is, so T2 would search a "
            "cloud that does not line up with the room")

    tests["frames"] = int(frames)
    tests["runtime_s"] = round(float(runtime_s), 1)
    ok_floor = frames >= criteria.min_frames and runtime_s >= criteria.min_runtime_s
    tests["past_minimums"] = ok_floor
    if not ok_floor:
        reasons.append(f"only {frames} frames in {runtime_s:.0f}s; minimums are "
                       f"{criteria.min_frames} and {criteria.min_runtime_s:.0f}s")

    ready = bool(ok_explored and ok_exits and ok_growth and ok_placed and ok_floor)
    return {
        "ready": ready,
        "tests": tests,
        "blocking": reasons,
        "recommendation": ("t1 is complete; switch to t2 and name the target"
                           if ready else "; ".join(reasons)),
    }
