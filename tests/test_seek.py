"""The decision stage: from a box in the image to a goal in the robot's map.

The same approach as ``test_compare.py``, for the same reason.  Every failure
this stage can have is silent — a coordinate that is confidently in the wrong
place looks exactly like one that is right — so the scenes here are synthetic,
with an answer computed by hand, and the assertions are about metres rather than
about the code having run.

The scene, used throughout: a camera at the origin looking down +x in the map
frame, a pointmap that is a flat wall 3 m ahead, and one object stuck to that
wall at a known pixel and a known height.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robocam.compare import CameraMount, cloud_to_map, map_to_cloud
from robocam.detect import Detection, queries_from_target
from robocam.occupancy import Grid
from robocam.odometry import Odom
from robocam.seek import (BASIS_LIVE, BASIS_MEMORY, REACH_NO_STANDING_ROOM,
                          REACH_OK, REACH_TOO_HIGH, REACH_UNKNOWN,
                          CoverageTracker, Keyframe, KeyframeStore, Placement,
                          ReachEnvelope, Sighting, T1Criteria, TargetMemory,
                          ViewGeometry, choose_approach, judge_reach, locate,
                          shrink_box, t1_exit_criteria)

W, H = 64, 48


def a_view(scale=1.0, crop_x0=0, crop_y0=0):
    return ViewGeometry(scale_x=scale, scale_y=scale, crop_x0=crop_x0, crop_y0=crop_y0,
                        out_w=W, out_h=H, src_w=int(W / scale) + crop_x0,
                        src_h=int(H / scale) + crop_y0)


def a_placement(x=0.0, y=0.0, yaw=0.0, map_id="m1"):
    """Camera at the robot's origin, no mount offset: cloud frame == optical frame."""
    return Placement(pose_c2w=np.eye(4),
                     odom=Odom(x=x, y=y, yaw=yaw, frame="map"),
                     mount=CameraMount(x=0.0, y=0.0, z=0.0),
                     map_id=map_id, cloud_map_id=1, frame="map", t_ns=1_000_000_000)


def a_scene(object_uv=(32, 24), object_depth=1.0, wall_depth=3.0, object_half=4,
            object_optical_y=0.0):
    """A wall at ``wall_depth`` with one nearer object patch stuck in front of it.

    Optical convention: x right, y down, z forward.  ``object_optical_y`` is
    therefore the object's height *below* the optical axis, which for a camera at
    z=0 in the map frame is a negative map-frame z.
    """
    pointmap = np.zeros((H, W, 3), dtype=np.float64)
    xs = (np.arange(W) - W / 2.0) / (W / 2.0)
    ys = (np.arange(H) - H / 2.0) / (H / 2.0)
    pointmap[:, :, 0] = xs[None, :]
    pointmap[:, :, 1] = ys[:, None]
    pointmap[:, :, 2] = wall_depth

    u, v = object_uv
    sl = (slice(v - object_half, v + object_half + 1),
          slice(u - object_half, u + object_half + 1))
    pointmap[sl][:, :, 2] = object_depth
    pointmap[sl][:, :, 1] = object_optical_y
    pointmap[sl][:, :, 0] = 0.0

    conf = np.full((H, W), 3.0)
    return pointmap, conf


def a_detection(uv=(32, 24), half=4, score=0.9):
    u, v = uv
    return Detection(box=(u - half, v - half, u + half + 1, v + half + 1),
                     score=score, label="mug", query="mug")


# -- the pixel -> pointmap mapping -------------------------------------------

def test_a_box_maps_through_the_crop_the_model_actually_applied():
    """The mapping that fails silently if it is wrong.

    A 1280-wide frame scaled by 0.4 and cropped 20 px from the left puts image
    column 100 at pointmap column 20.  Getting this wrong does not raise — it
    reads the depth of whatever is a few centimetres to the side.
    """
    view = ViewGeometry(scale_x=0.4, scale_y=0.4, crop_x0=20, crop_y0=10,
                        out_w=W, out_h=H, src_w=1280, src_h=720)
    assert view.image_to_pointmap(100, 100) == pytest.approx((20.0, 30.0))
    assert view.pointmap_to_image(20.0, 30.0) == pytest.approx((100.0, 100.0))


def test_a_box_entirely_outside_the_crop_is_none_not_an_empty_box():
    """"Outside the reconstruction" and "not there" are different answers.

    The crop is narrower than the camera's frame, so a target at the edge of the
    view has a box and no points.  The robot should turn towards it, not conclude
    the object is gone, and that only works if the two cases are distinguishable.
    """
    view = a_view()
    assert view.box_to_pointmap((500, 500, 520, 520)) is None
    assert view.box_to_pointmap((10, 10, 20, 20)) is not None


def test_shrink_box_keeps_the_middle():
    assert shrink_box((0, 0, 100, 100), keep=0.5) == (25, 25, 75, 75)


# -- locating ----------------------------------------------------------------

def test_the_object_lands_where_the_geometry_says_it_should():
    """A patch 1 m down the optical axis is 1 m in front of the robot."""
    pointmap, conf = a_scene(object_depth=1.0)
    sighting, why = locate(a_detection(), pointmap, conf, a_view(), a_placement(),
                           "mug", min_conf=1.5, min_points=4)
    assert sighting is not None, why
    assert sighting.x == pytest.approx(1.0, abs=1e-6)
    assert sighting.y == pytest.approx(0.0, abs=1e-6)
    assert sighting.range_m == pytest.approx(1.0, abs=1e-6)


def test_the_median_depth_filter_rejects_the_wall_behind_the_object():
    """The filter that decides whether the coordinate is usable at all.

    A bounding box always contains background — a detector draws a rectangle and
    an object is not one.  Here the box is a little larger than the object, so
    about a third of the pixels behind it are the wall 3 m away, and the naive
    centroid of everything lands near 1.8 m: between the object and the wall,
    pointing at nothing, and entirely plausible-looking.  Keeping only the
    dominant surface puts the coordinate on the object.
    """
    pointmap, conf = a_scene(object_depth=1.0, wall_depth=3.0, object_half=7)
    detection = a_detection(half=9)          # box a little larger than the object
    sighting, why = locate(detection, pointmap, conf, a_view(), a_placement(),
                           "mug", min_conf=1.5, min_points=4)
    assert sighting is not None, why
    naive = float(pointmap[detection.box[1]:detection.box[3],
                           detection.box[0]:detection.box[2], 2].mean())
    assert naive > 1.5, "the scene must actually pose the problem"
    assert sighting.range_m == pytest.approx(1.0, abs=0.05)


def test_a_box_over_unconfident_points_is_refused_rather_than_guessed():
    pointmap, conf = a_scene()
    conf[:] = 1.0                              # the model is sure about nothing
    sighting, why = locate(a_detection(), pointmap, conf, a_view(), a_placement(),
                           "mug", min_conf=1.5, min_points=4)
    assert sighting is None
    assert why["reason"] == "too_few_confident_points"


def test_a_box_outside_the_reconstruction_says_so():
    pointmap, conf = a_scene()
    detection = Detection(box=(900, 900, 950, 950), score=0.9)
    sighting, why = locate(detection, pointmap, conf, a_view(), a_placement(), "mug")
    assert sighting is None
    assert why["reason"] == "outside_reconstruction"


def test_the_robot_pose_carries_into_the_coordinate():
    """The same pixels, from a robot standing somewhere else, are somewhere else.

    A sighting is in the map frame, so it must move with the robot's pose.  A
    stage that returned camera-frame coordinates would pass every test above and
    send the robot to the wrong room.
    """
    pointmap, conf = a_scene(object_depth=1.0)
    here, _ = locate(a_detection(), pointmap, conf, a_view(), a_placement(),
                     "mug", min_points=4)
    there, _ = locate(a_detection(), pointmap, conf, a_view(),
                      a_placement(x=5.0, y=2.0, yaw=math.pi / 2), "mug", min_points=4)
    assert here.x == pytest.approx(1.0, abs=1e-6)
    # Turned 90° left at (5, 2): 1 m "ahead" is now +y.
    assert there.x == pytest.approx(5.0, abs=1e-6)
    assert there.y == pytest.approx(3.0, abs=1e-6)


def test_a_map_coordinate_round_trips_back_into_the_cloud():
    """The transform the goal coordinate travels on, in both directions.

    Once the robot has been told "the mug is at (x, y)", every later question
    about that place is a question about a region of the cloud, and the answer
    has to come back through the same transform it went out on.
    """
    placement = a_placement(x=1.5, y=-0.5, yaw=0.7)
    original = np.array([[0.3, -0.2, 2.4], [0.0, 0.0, 1.0]])
    back = placement.to_cloud(placement.to_map(original))
    assert back == pytest.approx(original, abs=1e-9)

    # And through the free functions, which is how compare.py exposes it.
    odom, mount = placement.odom, placement.mount
    in_map = cloud_to_map(original, placement.pose_c2w, odom, mount)
    assert map_to_cloud(in_map, placement.pose_c2w, odom, mount) == \
        pytest.approx(original, abs=1e-9)


# -- reachability ------------------------------------------------------------

def a_sighting(x=2.0, y=0.0, z=0.05, map_id="m1", **kw):
    return Sighting(target="mug", x=x, y=y, z=z, confidence=0.9, map_id=map_id, **kw)


def an_open_grid():
    """A 4 m square room with free interior — somewhere to stand everywhere."""
    cells = np.zeros((80, 80), dtype=np.int8)
    cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 100
    return Grid(cells=cells, resolution=0.05, origin=(0.0, 0.0, 0.0),
                frame="map", map_id="m1")


def test_an_object_on_the_floor_is_reachable():
    reach = judge_reach(a_sighting(z=0.05), ReachEnvelope(), an_open_grid(),
                        from_xy=(0.5, 0.0))
    assert reach.reachable is True
    assert reach.verdict == REACH_OK
    assert reach.approach is not None


def test_an_object_on_a_table_is_not():
    """The verdict that only the reconstruction can produce.

    The LiDAR sees one horizontal plane and cannot tell a mug on the floor from
    one on a table — the two are the same reading, or no reading at all.  Height
    is exactly what the cloud knows, and reporting it before the robot drives is
    the whole value of having one.
    """
    reach = judge_reach(a_sighting(z=0.75), ReachEnvelope(grasp_z_max=0.12),
                        an_open_grid(), from_xy=(0.5, 0.0))
    assert reach.reachable is False
    assert reach.verdict == REACH_TOO_HIGH
    # Still given an approach: knowing where the mug is has value even when it
    # cannot be picked up, and the robot may want to go and look at it.
    assert reach.approach is not None
    assert "0.75" in reach.reason


def test_the_boundary_is_where_the_config_puts_it():
    envelope = ReachEnvelope(grasp_z_max=0.12)
    grid, here = an_open_grid(), (0.5, 0.0)
    assert judge_reach(a_sighting(z=0.119), envelope, grid, here).reachable is True
    assert judge_reach(a_sighting(z=0.121), envelope, grid, here).reachable is False


def test_a_point_under_the_floor_is_a_reconstruction_error_not_an_object():
    reach = judge_reach(a_sighting(z=-0.4), ReachEnvelope(), an_open_grid(), (0.5, 0.0))
    assert reach.reachable is False
    assert reach.verdict == "below_floor"


def test_an_unmeasurable_height_is_unknown_rather_than_a_guess():
    """The third value, and why it is not folded into False.

    The robot's correct response to "unknown" is to drive over and look, which is
    different from its response to "no".  A two-valued answer would have to pick
    one of those and would be wrong half the time.
    """
    reach = judge_reach(a_sighting(z=0.0), ReachEnvelope(), an_open_grid(),
                        (0.5, 0.0), z_known=False)
    assert reach.reachable is None
    assert reach.verdict == REACH_UNKNOWN


def test_an_object_walled_in_on_every_side_has_nowhere_to_stand():
    cells = np.full((80, 80), 100, dtype=np.int8)   # solid: no free cell anywhere
    grid = Grid(cells=cells, resolution=0.05, origin=(0.0, 0.0, 0.0),
                frame="map", map_id="m1")
    reach = judge_reach(a_sighting(x=2.0, y=2.0), ReachEnvelope(), grid, (1.0, 2.0))
    assert reach.reachable is False
    assert reach.verdict == REACH_NO_STANDING_ROOM
    assert reach.approach is None


def test_a_blocked_direct_approach_goes_round_rather_than_giving_up():
    """A mug on the far side of a table: the natural approach is inside the table."""
    cells = np.zeros((80, 80), dtype=np.int8)
    # The object is at x=2.0 and the robot at x=1.0, so the direct standoff pose
    # is at x=1.2 -- column 24 at 5 cm/cell.  Put the table there.
    cells[:, 22:27] = 100
    grid = Grid(cells=cells, resolution=0.05, origin=(0.0, 0.0, 0.0),
                frame="map", map_id="m1")
    pose, note, detail = choose_approach(a_sighting(x=2.0, y=2.0), ReachEnvelope(),
                                         grid, from_xy=(1.0, 2.0))
    assert pose is not None
    assert detail["approach_source"] == "ring"
    assert "blocked" in note


def test_the_approach_faces_the_object():
    """The sign that decides whether the robot arrives facing it or away from it."""
    envelope = ReachEnvelope(standoff_m=0.8)
    pose, _, _ = choose_approach(a_sighting(x=3.0, y=0.0), envelope, None, (0.0, 0.0))
    px, py, yaw = pose
    assert math.hypot(3.0 - px, 0.0 - py) == pytest.approx(0.8, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)          # looking along +x, at it
    assert px < 3.0                                     # short of it, not past it


# -- memory ------------------------------------------------------------------

def test_a_remembered_coordinate_from_a_different_map_is_not_offered():
    """The check that is easy to omit and expensive to omit.

    After a SLAM reset the robot's map has new coordinates and a remembered
    (x, y) names a place that no longer exists — while looking exactly like a
    valid goal.
    """
    memory = TargetMemory()
    memory.remember(a_sighting(map_id="m1"))
    assert memory.recall("mug", map_id="m1") is not None
    assert memory.recall("mug", map_id="m2") is None


def test_a_new_mission_replaces_the_old_coordinate():
    memory = TargetMemory()
    memory.remember(a_sighting())
    other = Sighting(target="ball", x=9.0, y=9.0, z=0.0, confidence=0.1, map_id="m1")
    assert memory.remember(other) is True
    assert memory.recall("ball", map_id="m1").x == 9.0


def test_a_stale_memory_expires():
    memory = TargetMemory(ttl_s=10.0)
    old = a_sighting()
    old.t_ns = 1_000_000_000
    memory.remember(old)
    assert memory.recall("mug", now_ns=5_000_000_000) is not None
    assert memory.recall("mug", now_ns=60_000_000_000) is None


def test_keyframes_are_thinned_by_distance_not_by_time():
    """A parked robot produces identical frames and none of them is new evidence."""
    store = KeyframeStore(min_spacing_m=0.25, min_spacing_rad=10.0)

    def kf(x):
        placement = a_placement(x=x)
        return Keyframe(image=np.zeros((4, 4, 3), np.uint8), view=a_view(),
                        placement=placement, pointmap=np.zeros((4, 4, 3)),
                        conf=np.zeros((4, 4)))

    assert store.add(kf(0.0)) is True
    assert store.add(kf(0.05)) is False      # 5 cm: the same view
    assert store.add(kf(0.30)) is True
    assert len(store) == 2


def test_the_keyframe_store_is_bounded():
    store = KeyframeStore(max_frames=3, min_spacing_m=0.0, min_spacing_rad=0.0)
    for i in range(10):
        store.add(Keyframe(image=np.zeros((2, 2, 3), np.uint8), view=a_view(),
                           placement=a_placement(x=float(i)),
                           pointmap=np.zeros((2, 2, 3)), conf=np.zeros((2, 2)), seq=i))
    assert len(store) == 3
    assert [k.seq for k in store.newest_first()] == [9, 8, 7]


def test_recall_searches_newest_first():
    store = KeyframeStore(min_spacing_m=0.0, min_spacing_rad=0.0)
    for i in range(3):
        store.add(Keyframe(image=np.zeros((2, 2, 3), np.uint8), view=a_view(),
                           placement=a_placement(x=float(i)),
                           pointmap=np.zeros((2, 2, 3)), conf=np.zeros((2, 2)), seq=i))
    assert store.newest_first()[0].seq == 2


# -- when is T1 finished? ----------------------------------------------------

def a_grid_explored(fraction: float) -> Grid:
    cells = np.full((100, 100), -1, dtype=np.int8)
    observed = int(fraction * cells.size)
    cells.reshape(-1)[:observed] = 0
    grid = Grid(cells=cells, resolution=0.05, origin=(0.0, 0.0, 0.0),
                frame="map", map_id="m1")
    grid.summary = {"explored_fraction": fraction, "unknown": cells.size - observed}
    return grid


def a_stalled_coverage() -> CoverageTracker:
    tracker = CoverageTracker(window_s=30.0)
    tracker.observe(9000, now=0.0)
    tracker.observe(9010, now=40.0)      # 0.25 cells/s: not growing
    return tracker


def finished_kwargs(**overrides):
    kwargs = dict(
        grid=a_grid_explored(0.9),
        exits=[],
        criteria=T1Criteria(),
        coverage=a_stalled_coverage(),
        mean_agreement=0.5,
        frames=1000,
        runtime_s=300.0,
    )
    kwargs.update(overrides)
    return kwargs


def test_a_finished_t1_says_so():
    kwargs = finished_kwargs()
    verdict = t1_exit_criteria(kwargs.pop("grid"), kwargs.pop("exits"), **kwargs)
    assert verdict["ready"] is True
    assert verdict["blocking"] == []


@pytest.mark.parametrize("override,expected_test", [
    ({"grid": a_grid_explored(0.4)}, "explored"),
    ({"exits": [{"status": "open", "width_m": 0.9}]}, "no_open_exits"),
    ({"mean_agreement": 0.01}, "placed"),
    ({"mean_agreement": None}, "placed"),
    ({"frames": 3}, "past_minimums"),
])
def test_each_test_can_block_the_transition_on_its_own(override, expected_test):
    kwargs = finished_kwargs(**override)
    verdict = t1_exit_criteria(kwargs.pop("grid"), kwargs.pop("exits"), **kwargs)
    assert verdict["ready"] is False
    assert verdict["tests"][expected_test] is False
    assert verdict["blocking"], "a blocked transition must say what is blocking it"


def test_a_map_still_growing_is_not_finished_however_well_explored():
    """The test that catches the case the explored fraction cannot.

    A robot halfway down a long corridor can have observed most of its grid and
    still be discovering a room a second.
    """
    growing = CoverageTracker(window_s=30.0)
    growing.observe(1000, now=0.0)
    growing.observe(9000, now=40.0)          # 200 cells/s
    kwargs = finished_kwargs(coverage=growing)
    verdict = t1_exit_criteria(kwargs.pop("grid"), kwargs.pop("exits"), **kwargs)
    assert verdict["ready"] is False
    assert verdict["tests"]["not_growing"] is False


def test_a_narrow_frontier_does_not_block_the_transition():
    """A 40 cm gap is noise between two obstacle cells, not a door this robot fits."""
    kwargs = finished_kwargs(exits=[{"status": "open", "width_m": 0.3}])
    verdict = t1_exit_criteria(kwargs.pop("grid"), kwargs.pop("exits"),
                               min_exit_width_m=0.6, **kwargs)
    assert verdict["tests"]["open_exits"] == 0
    assert verdict["ready"] is True


def test_never_having_compared_reads_differently_from_having_disagreed():
    """None and 0.0 are different failures with different fixes.

    Never compared is a configuration problem — no grid, no pose, compare off.
    Compared and agreed with nothing is a placement problem — a wrong camera
    height, a wrong mount, a pose in the wrong frame.
    """
    kwargs = finished_kwargs(mean_agreement=None)
    never = t1_exit_criteria(kwargs.pop("grid"), kwargs.pop("exits"), **kwargs)
    kwargs = finished_kwargs(mean_agreement=0.0)
    disagreed = t1_exit_criteria(kwargs.pop("grid"), kwargs.pop("exits"), **kwargs)
    assert "never been compared" in " ".join(never["blocking"])
    assert "agreement is 0.00" in " ".join(disagreed["blocking"])


def test_the_coverage_tracker_needs_history_before_it_answers():
    tracker = CoverageTracker()
    assert tracker.rate_cells_per_s() is None
    tracker.observe(100, now=0.0)
    assert tracker.rate_cells_per_s() is None
    tracker.observe(200, now=10.0)
    assert tracker.rate_cells_per_s() == pytest.approx(10.0)
