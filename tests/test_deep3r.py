"""Tests for the deep3r processor that need neither a GPU nor CUT3R.

Everything below the model — the confidence filter, the voxel reduction, the
quantisation, and the state machine that decides when a map ends — is plain
numpy, and it is where the bugs that corrupt a map quietly live.  Keeping it
testable without weights is what lets those be checked on a laptop, in the
default venv, which has no torch at all.
"""

import threading
import time

import numpy as np
import pytest

from robocam.processors import REGISTRY, build
from robocam.processors.base import Frame
from robocam.processors.deep3r import Deep3RProcessor


def make_proc(**options) -> Deep3RProcessor:
    """A processor that is never set up, so nothing imports torch."""
    return Deep3RProcessor(**options)


def stub_model(proc, points=64, conf=5.0):
    """Replace the model with a fixed cloud so process() can be exercised."""
    calls = []

    def _run_model(image, reset):
        calls.append(reset)
        # The real one establishes the recurrent state; without this the
        # processor would see a None state and reset on every single frame.
        proc._state = ("stubbed",)
        pts = np.random.default_rng(0).uniform(-1, 1, size=(points, 3))
        rgb = np.zeros((points, 3), np.uint8)
        return pts, rgb, np.full(points, conf), np.eye(4)

    proc._run_model = _run_model
    return calls


def frame(seq: int) -> Frame:
    return Frame(seq=seq, session_id="s", image=np.zeros((48, 64, 3), np.uint8))


# -- registry ----------------------------------------------------------------


def test_deep3r_is_registered():
    assert "deep3r" in REGISTRY
    assert isinstance(build("deep3r", {}), Deep3RProcessor)


def test_importing_the_registry_does_not_import_torch():
    """The whole point of the lazy import in setup().

    ``robocam.processors`` imports every registered processor at startup, so a
    module-level torch import here would stop a server that only ever wanted
    ``stats`` from starting on a machine with no CUDA.

    Checked in a fresh interpreter rather than against this one's
    ``sys.modules``: in the venv that *has* torch, any earlier test importing
    it would make the in-process version of this check pass for the wrong
    reason, or fail for one.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    code = (
        "import sys; import robocam.processors; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=root)
    assert done.returncode == 0, "importing robocam.processors pulled in torch"


# -- voxel reduction ---------------------------------------------------------


def test_voxel_downsample_keeps_the_most_confident_point_per_voxel():
    proc = make_proc(voxel_m=1.0)
    pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [5.5, 5.5, 5.5]])
    conf = np.array([1.0, 9.0, 4.0])
    rgb = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], np.uint8)

    kept_pts, kept_rgb, kept_conf = proc._voxel_downsample(pts, rgb, conf)

    assert kept_pts.shape[0] == 2
    # Averaging would smear the surface across the noise sharing the cell;
    # the winner must be the point the model was actually sure about.
    assert 9.0 in kept_conf and 1.0 not in kept_conf
    assert [2, 2, 2] in kept_rgb.tolist()


def test_voxel_downsample_collapses_a_dense_plane():
    proc = make_proc(voxel_m=0.1)
    grid = np.mgrid[0:20, 0:20].reshape(2, -1).T * 0.01
    pts = np.column_stack([grid, np.zeros(len(grid))])
    conf = np.arange(len(pts), dtype=float)

    kept, _, _ = proc._voxel_downsample(pts, None, conf)

    # 20x20 points spanning 0.19 m at a 0.1 m voxel is a 2x2 grid of cells.
    assert kept.shape[0] == 4


def test_voxel_downsample_tolerates_missing_colours():
    proc = make_proc(voxel_m=1.0)
    pts = np.zeros((3, 3))
    _, rgb, _ = proc._voxel_downsample(pts, None, np.ones(3))
    assert rgb is None


# -- cloud encoding ----------------------------------------------------------


def decode(cloud):
    """Undo the wire encoding exactly as the robot must."""
    import base64

    raw = np.frombuffer(base64.b64decode(cloud["xyz_u16"]), "<u2").reshape(-1, 3)
    return np.asarray(cloud["origin"]) + cloud["scale"] * raw


def test_encode_cloud_round_trips_within_quantisation_error():
    proc = make_proc(voxel_m=0.001, max_points=10_000, min_conf=0.0)
    rng = np.random.default_rng(1)
    pts = rng.uniform(-5, 5, size=(500, 3))
    conf = np.full(500, 9.0)

    cloud = proc._encode_cloud(pts, None, conf)
    back = decode(cloud)

    # Voxelisation reorders the points, so "round trips" means every decoded
    # point coincides with an original one -- not that index i matches index i.
    assert back.shape == pts.shape
    nearest = np.linalg.norm(back[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
    # 16 bits over a 10 m extent is a fifth of a millimetre, far finer than
    # anything the model itself resolves.
    assert nearest.max() < 1e-3


def test_encode_cloud_drops_low_confidence_points():
    proc = make_proc(min_conf=2.0, voxel_m=0.001, max_points=10_000)
    pts = np.arange(30, dtype=float).reshape(10, 3)
    conf = np.linspace(1.0, 3.0, 10)

    cloud = proc._encode_cloud(pts, None, conf)

    assert cloud["n_raw"] == 10
    assert cloud["n_after_conf"] == int((conf >= 2.0).sum())


def test_encode_cloud_drops_non_finite_points():
    """A NaN reaching a costmap is an obstacle at an undefined place."""
    proc = make_proc(min_conf=0.0, voxel_m=0.001, max_points=10_000)
    pts = np.array([[0.0, 0.0, 1.0], [np.nan, 0.0, 1.0], [0.0, np.inf, 1.0]])

    cloud = proc._encode_cloud(pts, None, np.full(3, 9.0))

    assert cloud["n_after_conf"] == 1


def test_encode_cloud_caps_by_confidence_not_at_random():
    proc = make_proc(min_conf=0.0, voxel_m=1e-6, max_points=5)
    pts = np.arange(300, dtype=float).reshape(100, 3)
    conf = np.arange(100, dtype=float)

    cloud = proc._encode_cloud(pts, None, conf)
    back = decode(cloud)

    assert cloud["n_points"] == 5
    # Dropping at random would thin the surfaces Nav2 needs along with the
    # noise, so the survivors must be the five most confident points.
    assert set(back[:, 0].round().astype(int)) == {285, 288, 291, 294, 297}


def test_encode_cloud_survives_an_empty_cloud():
    """Everything filtered out is a normal frame, not an error."""
    proc = make_proc(min_conf=9.0)
    cloud = proc._encode_cloud(np.zeros((4, 3)), None, np.ones(4))

    assert cloud["n_points"] == 0
    assert cloud["xyz_u16"] == ""
    assert cloud["scale"] == 0.0


def test_encode_cloud_handles_a_single_point():
    """A zero extent must not divide by zero on the way to the wire."""
    proc = make_proc(min_conf=0.0, voxel_m=1.0)
    cloud = proc._encode_cloud(np.array([[1.0, 2.0, 3.0]]), None, np.array([9.0]))

    assert cloud["n_points"] == 1
    assert np.allclose(decode(cloud)[0], [1.0, 2.0, 3.0])


# -- state machine -----------------------------------------------------------


def test_first_frame_starts_a_map():
    proc = make_proc()
    stub_model(proc)

    out = proc.process(frame(0))

    assert out["reset"] is True
    assert out["reset_reason"] == "first frame"
    assert out["map_id"] == 1
    assert out["frames_in_state"] == 1


def test_consecutive_frames_extend_one_map():
    proc = make_proc()
    stub_model(proc)

    outs = [proc.process(frame(i)) for i in range(4)]

    assert [o["map_id"] for o in outs] == [1, 1, 1, 1]
    assert [o["frames_in_state"] for o in outs] == [1, 2, 3, 4]
    assert [o["reset"] for o in outs[1:]] == [False, False, False]


def test_sequence_gap_starts_a_new_map():
    """A long gap means the state no longer overlaps what the camera sees.

    Fusing across it would weld two unrelated scenes into one coordinate
    frame, which is worse than admitting the map ended.
    """
    proc = make_proc(reset_on_gap=5)
    stub_model(proc)
    proc.process(frame(0))

    out = proc.process(frame(50))

    assert out["reset"] is True
    assert "gap" in out["reset_reason"]
    assert out["seq_gap"] == 50
    assert out["map_id"] == 2


def test_a_small_gap_does_not_start_a_new_map():
    proc = make_proc(reset_on_gap=5)
    stub_model(proc)
    proc.process(frame(0))

    out = proc.process(frame(3))

    assert out["reset"] is False
    assert out["map_id"] == 1


def test_reset_every_bounds_the_state():
    proc = make_proc(reset_every=3, reset_on_gap=1000)
    stub_model(proc)

    ids = [proc.process(frame(i))["map_id"] for i in range(7)]

    # Three frames per map, then a fresh one: the state cannot grow forever.
    assert ids == [1, 1, 1, 2, 2, 2, 3]


def test_every_n_skips_without_running_the_model():
    proc = make_proc(every_n=3, reset_on_gap=1000)
    calls = stub_model(proc)

    outs = [proc.process(frame(i)) for i in range(6)]

    statuses = [o.get("status") for o in outs]
    assert statuses == [None, "skipped", "skipped", None, "skipped", "skipped"]
    # Skipping in the processor beats letting the queue evict, because the
    # frames that survive are then chosen rather than whichever happened to
    # arrive between forward passes.
    assert len(calls) == 2


def test_a_skipped_frame_still_returns_a_result():
    """The client counts replies to decide when to send, so silence stalls it."""
    proc = make_proc(every_n=2)
    stub_model(proc)
    proc.process(frame(0))

    out = proc.process(frame(1))

    assert out["status"] == "skipped"
    assert "map_id" in out


def test_reset_on_gap_defaults_to_a_multiple_of_every_n():
    """The threshold has to scale with the skipping, or it fires constantly."""
    assert make_proc(every_n=5).reset_on_gap == 20
    assert make_proc(every_n=5, reset_on_gap=7).reset_on_gap == 7


# -- concurrency -------------------------------------------------------------


def test_a_second_worker_is_refused():
    """Two workers stepping one recurrent state corrupt the map in silence.

    There is no output that would look wrong, so this has to fail loudly
    rather than be left to the config being right.
    """
    proc = make_proc()
    stub_model(proc)
    proc._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="workers = 1"):
            proc.process(frame(0))
    finally:
        proc._lock.release()


def test_the_lock_is_released_when_the_model_raises():
    proc = make_proc()

    def _boom(image, reset):
        raise ValueError("model exploded")

    proc._run_model = _boom
    with pytest.raises(ValueError):
        proc.process(frame(0))

    # A processor raising is not fatal to the server, so the next frame has to
    # be able to run rather than inherit a lock the failure never gave back.
    assert proc._lock.acquire(blocking=False)
    proc._lock.release()


def test_two_threads_are_never_inside_the_model_at_once():
    """The invariant the guard exists for, asserted from inside the model.

    Counting outcomes afterwards cannot distinguish "serialised correctly"
    from "both ran and one result was lost", so the check belongs where the
    overlap would happen.
    """
    proc = make_proc(reset_on_gap=1000)
    inside = []
    overlaps = []

    def _run_model(image, reset):
        inside.append(1)
        if len(inside) > 1:
            overlaps.append(len(inside))
        time.sleep(0.02)
        inside.pop()
        proc._state = ("stubbed",)
        return (np.zeros((4, 3)), None, np.full(4, 9.0), np.eye(4))

    proc._run_model = _run_model
    refused = []

    def worker(seq):
        try:
            proc.process(frame(seq))
        except RuntimeError:  # the guard doing its job, not a test failure
            refused.append(seq)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert overlaps == []
    # With four threads racing a non-blocking guard, at least one must have
    # been turned away -- otherwise the test proved nothing.
    assert refused


# -- scale check -------------------------------------------------------------


class FakeScan:
    def __init__(self, **summary):
        self.seq = 1
        self.summary = summary


def test_scale_check_compares_the_cloud_with_the_lidar():
    proc = make_proc(min_conf=1.0)
    f = frame(0)
    f.scan = FakeScan(front_min_m=2.0)
    pts = np.tile([0.0, 0.0, 2.0], (100, 1))

    out = proc._scale_check(f, pts, np.full(100, 5.0))

    assert out["lidar_front_m"] == 2.0
    # CUT3R claims metric scale and nothing else in the pipeline would notice
    # if it were wrong by a factor of two; this is the cheapest check of it.
    assert out["ratio"] == pytest.approx(1.0, abs=0.01)


def test_scale_check_falls_back_to_the_nearest_return():
    """front_min_m is None in open space, which is a reading, not a fault."""
    proc = make_proc(min_conf=1.0)
    f = frame(0)
    f.scan = FakeScan(front_min_m=None, nearest_m=4.0)
    pts = np.tile([0.0, 0.0, 2.0], (10, 1))

    out = proc._scale_check(f, pts, np.full(10, 5.0))

    assert out["lidar_front_m"] == 4.0
    assert out["ratio"] == pytest.approx(0.5, abs=0.01)


def test_scale_check_is_absent_without_a_scan():
    proc = make_proc()
    assert proc._scale_check(frame(0), np.zeros((3, 3)), np.ones(3)) is None


def test_scale_check_is_absent_when_the_scan_saw_nothing():
    proc = make_proc()
    f = frame(0)
    f.scan = FakeScan(front_min_m=None, nearest_m=None)
    assert proc._scale_check(f, np.zeros((3, 3)), np.ones(3)) is None


# -- the comparison against the robot's map ----------------------------------
#
# The production path, exercised without torch: _compare_with_map takes points
# and a pose and needs no model to do so.  The stub processor in test_loopback
# covers the server's half of the same journey; this covers deep3r's.


def a_configured_proc(**overrides):
    """A processor with the server config applied, as configure() would."""
    from robocam.config import Config

    cfg = Config()
    for section, values in overrides.items():
        for key, value in values.items():
            setattr(getattr(cfg, section), key, value)
    proc = make_proc()
    proc.configure(cfg.lidar, cfg.imu, cfg)
    return proc


def a_mapped_room(width=80, height=80):
    from robocam.occupancy import Grid

    cells = np.zeros((height, width), dtype=np.int8)
    cells[0, :] = 100
    cells[-1, :] = 100
    cells[:, 0] = 100
    cells[:, -1] = 100
    return Grid(cells=cells, resolution=0.05, origin=(0.0, 0.0, 0.0),
                frame="map", map_id="m1")


def a_frame_with_a_map(**kwargs):
    from robocam.odometry import Odom

    f = frame(0)
    f.grid = a_mapped_room()
    f.odom = Odom(x=0.5, y=2.0, yaw=0.0, frame="map")
    for key, value in kwargs.items():
        setattr(f, key, value)
    return f


def a_table_in_optical_coordinates(f, proc):
    """A 40 cm table top at 75 cm, expressed in the model's own frame."""
    import math

    xs, ys = np.meshgrid(np.linspace(1.8, 2.2, 20), np.linspace(1.8, 2.2, 20))
    table = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.75)])

    mount = proc._mount.matrix()
    c, s = math.cos(f.odom.yaw), math.sin(f.odom.yaw)
    t_map_base = np.eye(4)
    t_map_base[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t_map_base[:3, 3] = (f.odom.x, f.odom.y, f.odom.z)
    t_map_cam = t_map_base @ mount
    return (table - t_map_cam[:3, 3]) @ t_map_cam[:3, :3]


def test_configure_takes_the_camera_mount_from_the_server_config():
    """It describes the robot, not the model, so it must not live in options."""
    proc = a_configured_proc(compare={"camera_z": 0.62, "camera_pitch": 0.1})
    assert proc._mount.z == 0.62
    assert proc._mount.pitch == 0.1


def test_a_processor_configured_with_nothing_keeps_working():
    """configure(None) is what every test and every old caller passes."""
    proc = make_proc()
    proc.configure(None, None, None)
    assert proc._compare_cfg is None
    assert proc._compare_with_map(a_frame_with_a_map(), np.zeros((1, 3)),
                                  np.eye(4))["ran"] is False


def test_the_comparison_says_why_it_could_not_run():
    """Three situations that produce the same silence, told apart in one field.

    Without this, "no map on the server", "no pose to place the cloud with" and
    "compared and found nothing" are indistinguishable from the robot.
    """
    proc = a_configured_proc()

    no_map = a_frame_with_a_map()
    no_map.grid = None
    assert "occupancy grid" in proc._compare_with_map(no_map, np.zeros((1, 3)),
                                                     np.eye(4))["why"]

    no_pose = a_frame_with_a_map()
    no_pose.odom = None
    assert "pose" in proc._compare_with_map(no_pose, np.zeros((1, 3)),
                                            np.eye(4))["why"]

    empty = a_frame_with_a_map()
    assert "confidence filter" in proc._compare_with_map(
        empty, np.zeros((0, 3)), np.eye(4))["why"]


def test_the_comparison_announces_a_patch_for_a_table_the_scanner_missed():
    """deep3r's half of the map loop, end to end and without a model."""
    proc = a_configured_proc(compare={"min_points": 1, "camera_z": 0.5,
                                      "camera_x": 0.0})
    f = a_frame_with_a_map()
    points = a_table_in_optical_coordinates(f, proc)

    stats = proc._compare_with_map(f, points, np.eye(4))

    assert stats["ran"] is True
    assert stats["new_cells"] > 20
    assert len(f.announcements) == 1
    header, payload = f.announcements[0]
    assert header["type"] == "map_update"
    assert header["map_id"] == "m1"
    assert header["merge"] == "max"
    assert header["cells_changed"] == stats["new_cells"]
    # seq is left at 0 for the server to stamp: the counters are per session and
    # a worker thread has no business allocating them.
    assert header["seq"] == 0
    assert len(payload) > 0


def test_nothing_is_announced_when_the_frames_do_not_match():
    """The refusal, from inside the processor rather than only in compare()."""
    from robocam.odometry import Odom

    proc = a_configured_proc(compare={"min_points": 1})
    f = a_frame_with_a_map()
    f.odom = Odom(x=0.5, y=2.0, yaw=0.0, frame="odom")
    points = np.tile([0.0, 0.0, 2.0], (50, 1))

    stats = proc._compare_with_map(f, points, np.eye(4))

    assert stats["ran"] is False
    assert "refusing to compare across frames" in stats["error"]
    assert f.announcements == []


def test_no_hint_is_announced_when_hints_are_off():
    proc = a_configured_proc(compare={"min_points": 1, "camera_z": 0.5,
                                      "camera_x": 0.0, "send_hints": False})
    f = a_frame_with_a_map()
    proc._compare_with_map(f, a_table_in_optical_coordinates(f, proc), np.eye(4))
    assert all(h["type"] != "pose_hint" for h, _ in f.announcements)


def test_the_cloud_carries_a_pointcloud2_layout():
    """So the robot need not hardcode one that could then drift out of step."""
    proc = make_proc(colors=True)
    pts = np.random.default_rng(0).uniform(-1, 1, size=(30, 3))
    cloud = proc._encode_cloud(pts, np.zeros((30, 3), np.uint8), np.full(30, 5.0))

    pc2 = cloud["pc2"]
    assert pc2["height"] == 1 and pc2["width"] == cloud["n_points"]
    assert pc2["point_step"] == 16
    assert [f["name"] for f in pc2["fields"]] == ["x", "y", "z", "rgb"]
    assert [f["offset"] for f in pc2["fields"]] == [0, 4, 8, 12]
    # is_dense, because non-finite points were filtered out rather than kept.
    assert pc2["is_dense"] is True


def test_a_colourless_cloud_declares_a_shorter_point():
    proc = make_proc(colors=False)
    pts = np.random.default_rng(0).uniform(-1, 1, size=(30, 3))
    cloud = proc._encode_cloud(pts, None, np.full(30, 5.0))
    assert cloud["pc2"]["point_step"] == 12
    assert [f["name"] for f in cloud["pc2"]["fields"]] == ["x", "y", "z"]


# -- the decision stage ------------------------------------------------------
#
# The same trick as the comparison above: _decide takes a pointmap and a pose
# and needs no model, so T2's whole journey -- detect, locate, judge, announce --
# runs in the default venv with no weights and no card.


class StubDetector:
    """A detector that reports a fixed box, or nothing.

    Standing in for OWLv2 so that what is being tested is the stage rather than
    the model: everything interesting about T2 happens *after* the box.
    """

    name = "stub"
    open_vocabulary = True

    def __init__(self, box=(28, 20, 37, 29), score=0.9):
        self.box, self.score = box, score
        self.calls = []

    def detect(self, image, queries):
        from robocam.detect import Detection

        self.calls.append(list(queries))
        if self.box is None:
            return []
        return [Detection(box=self.box, score=self.score, label="mug",
                          query=queries[0] if queries else "")]

    def close(self):
        pass


PM_W, PM_H = 64, 48


def a_seeking_proc(detector=None, **overrides):
    """A processor configured for t2, with the detector already in place."""
    from robocam.seek import ViewGeometry

    proc = a_configured_proc(**overrides)
    proc._detector = detector if detector is not None else StubDetector()
    # setup() would have recorded this from the crop it actually applied; with no
    # model there is no crop, so the pointmap is the frame.
    proc._view_geometry = ViewGeometry(scale_x=1.0, scale_y=1.0, crop_x0=0, crop_y0=0,
                                       out_w=PM_W, out_h=PM_H, src_w=PM_W, src_h=PM_H)
    return proc


def a_pointmap(object_depth=1.0, object_height=0.05, camera_z=0.45,
               wall_depth=4.0, half=6):
    """A wall, with one object patch in front of it at a known height.

    Built in the model's optical frame (x right, y down, z forward) with the
    camera at ``camera_z`` above the floor, so an object at ``object_height``
    above the floor is ``camera_z - object_height`` *below* the optical axis.
    """
    pointmap = np.zeros((PM_H, PM_W, 3), dtype=np.float64)
    pointmap[:, :, 2] = wall_depth
    pointmap[:, :, 1] = camera_z          # the wall's points, at floor level
    u, v = PM_W // 2, PM_H // 2
    sl = (slice(v - half, v + half + 1), slice(u - half, u + half + 1))
    pointmap[sl][:, :, 2] = object_depth
    pointmap[sl][:, :, 1] = camera_z - object_height
    pointmap[sl][:, :, 0] = 0.0
    conf = np.full((PM_H, PM_W), 5.0)
    return pointmap.reshape(-1, 3), conf.reshape(-1)


def a_t2_frame(target="the red mug", **kwargs):
    f = a_frame_with_a_map(**kwargs)
    f.image = np.zeros((PM_H, PM_W, 3), np.uint8)
    f.phase = "t2"
    f.mission = {"target": target}
    return f


def test_t1_keeps_keyframes_and_announces_nothing():
    """The half of the mission that cannot be done live.

    T1 explores before the target has been named, so the only thing it can
    usefully do about a target is remember what it saw.
    """
    proc = a_seeking_proc()
    f = a_frame_with_a_map()
    f.image = np.zeros((PM_H, PM_W, 3), np.uint8)
    points, conf = a_pointmap()

    stats = proc._decide(f, points, conf, np.eye(4))

    assert stats["ran"] is False
    assert "t1" in stats["why"]
    assert stats["kept_keyframe"] is True
    assert f.announcements == []


def test_t2_without_a_target_says_so_rather_than_searching_for_nothing():
    proc = a_seeking_proc()
    f = a_t2_frame(target="")
    points, conf = a_pointmap()
    stats = proc._decide(f, points, conf, np.eye(4))
    assert stats["ran"] is False
    assert "no target" in stats["why"]


def test_a_disabled_stage_and_a_missing_detector_are_different_answers():
    """Both produce no `found`, and they have different fixes."""
    from robocam.config import Config

    cfg = Config()
    cfg.seek.enabled = False
    off = make_proc()
    off.configure(cfg.lidar, cfg.imu, cfg)
    assert "disabled" in off._decide(a_t2_frame(), *a_pointmap(), np.eye(4))["why"]

    missing = a_configured_proc()
    missing._detector = None
    assert "no detector" in missing._decide(a_t2_frame(), *a_pointmap(), np.eye(4))["why"]


def test_a_found_carries_a_map_frame_coordinate_and_an_approach():
    """T2 end to end: a box becomes a goal the robot can drive to."""
    proc = a_seeking_proc(compare={"camera_z": 0.45, "camera_x": 0.0},
                          seek={"detect_every_n": 1})
    f = a_t2_frame()
    points, conf = a_pointmap(object_depth=1.0, object_height=0.05)

    stats = proc._decide(f, points, conf, np.eye(4))

    assert stats["ran"] is True
    assert len(f.announcements) == 1
    header, _ = f.announcements[0]
    assert header["type"] == "found"
    assert header["found"] is True
    assert header["frame"] == "map"
    assert header["map_id"] == "m1"
    assert header["basis"] == "live"
    # The robot is at (0.5, 2.0) looking down +x, the object 1 m ahead of the
    # camera, which sits at the base origin in this config.
    assert header["x"] == pytest.approx(1.5, abs=0.05)
    assert header["y"] == pytest.approx(2.0, abs=0.05)
    assert header["z"] == pytest.approx(0.05, abs=0.03)
    # seq is left for the server to stamp, as with every other announcement.
    assert header["seq"] == 0
    approach = header["approach"]
    assert approach["x"] < header["x"], "the approach stops short of the object"


def test_an_object_on_the_floor_is_reported_reachable():
    proc = a_seeking_proc(compare={"camera_z": 0.45, "camera_x": 0.0},
                          seek={"detect_every_n": 1, "grasp_z_max": 0.12})
    f = a_t2_frame()
    proc._decide(f, *a_pointmap(object_height=0.05), np.eye(4))
    header, _ = f.announcements[0]
    assert header["reachable"] is True
    assert header["reach"]["verdict"] == "reachable"


def test_an_object_on_a_table_is_reported_unreachable_before_the_robot_drives():
    """The verdict the LiDAR cannot produce, arriving before the drive rather
    than after a failed grasp."""
    proc = a_seeking_proc(compare={"camera_z": 0.45, "camera_x": 0.0},
                          seek={"detect_every_n": 1, "grasp_z_max": 0.12})
    f = a_t2_frame()
    proc._decide(f, *a_pointmap(object_height=0.75), np.eye(4))
    header, _ = f.announcements[0]
    assert header["reachable"] is False
    assert header["reach"]["verdict"] == "too_high"
    # Still a found, and still with an approach: knowing where it is has value
    # even when it cannot be picked up.
    assert header["found"] is True
    assert header["approach"]["x"] == pytest.approx(header["x"] - 0.8, abs=0.05)


def test_the_target_text_reaches_the_detector_verbatim():
    """It is a prompt, not a class label; constraining it would throw away the
    half of the description that makes the object findable."""
    detector = StubDetector()
    proc = a_seeking_proc(detector, seek={"detect_every_n": 1})
    proc._decide(a_t2_frame(target="the red mug on the desk"), *a_pointmap(), np.eye(4))
    assert detector.calls[0][0] == "the red mug on the desk"


def test_nothing_seen_and_nothing_remembered_eventually_says_so():
    """"I have looked and it is not here" is what lets the robot stop waiting on
    its monitor branch and start searching."""
    proc = a_seeking_proc(StubDetector(box=None),
                          seek={"detect_every_n": 1, "absent_after_runs": 3})
    announcements = []
    for _ in range(3):
        f = a_t2_frame()
        proc._decide(f, *a_pointmap(), np.eye(4))
        announcements.extend(f.announcements)

    assert len(announcements) == 1
    header, _ = announcements[0]
    assert header["type"] == "found"
    assert header["found"] is False
    assert header["basis"] == "absent"
    assert header["reachable"] is None


def test_a_low_scoring_detection_is_not_acted_on():
    proc = a_seeking_proc(StubDetector(score=0.05),
                          seek={"detect_every_n": 1, "min_confidence": 0.25})
    f = a_t2_frame()
    stats = proc._decide(f, *a_pointmap(), np.eye(4))
    assert f.announcements == []
    assert "min_confidence" in stats["rejected"]


def test_founds_are_rate_limited_because_the_robot_acts_on_them():
    proc = a_seeking_proc(seek={"detect_every_n": 1,
                                "min_found_interval_ms": 60_000.0,
                                "resend_move_m": 10.0})
    seen = 0
    for _ in range(5):
        f = a_t2_frame()
        proc._decide(f, *a_pointmap(), np.eye(4))
        seen += len(f.announcements)
    assert seen == 1


def test_a_target_that_has_moved_is_news_inside_the_interval():
    proc = a_seeking_proc(seek={"detect_every_n": 1,
                                "min_found_interval_ms": 60_000.0,
                                "resend_move_m": 0.3})
    first = a_t2_frame()
    proc._decide(first, *a_pointmap(object_depth=1.0), np.eye(4))
    moved = a_t2_frame()
    proc._decide(moved, *a_pointmap(object_depth=2.0), np.eye(4))
    assert len(first.announcements) == 1
    assert len(moved.announcements) == 1


def test_a_target_named_at_t2_is_looked_for_in_what_t1_saw():
    """The mission's shape, in one test.

    T1 explores without knowing what it will be asked for.  The target is named
    at T2 launch, and the answer already exists in the keyframes T1 kept -- so the
    robot gets a goal coordinate before T2 has taken a single frame of its own.
    """
    detector = StubDetector()
    proc = a_seeking_proc(detector, compare={"camera_z": 0.45, "camera_x": 0.0},
                          seek={"detect_every_n": 1})
    points, conf = a_pointmap(object_depth=1.0, object_height=0.05)

    # T1: drive along, keeping keyframes.  No target is named yet.
    for i in range(4):
        f = a_frame_with_a_map()
        f.image = np.zeros((PM_H, PM_W, 3), np.uint8)
        f.odom.x = 0.5 + i          # far enough apart to each be kept
        proc._decide(f, points, conf, np.eye(4))
    assert len(proc._keyframes) == 4

    # T2 begins, and the target is named for the first time.  This frame's own
    # view is of nothing: only the memory can answer.
    detector.box = None
    live_miss = np.zeros_like(points)
    live_conf = np.zeros_like(conf)
    f = a_t2_frame(target="the red mug")

    def only_in_keyframes(image, queries):
        from robocam.detect import Detection
        # Blank frames are the live ones; the stored keyframes are not blank.
        if not image.any():
            return []
        return [Detection(box=(28, 20, 37, 29), score=0.9, label="mug")]

    detector.detect = only_in_keyframes
    f.image = np.zeros((PM_H, PM_W, 3), np.uint8)
    for kf in proc._keyframes.newest_first():
        kf.image = np.ones((PM_H, PM_W, 3), np.uint8)

    stats = proc._decide(f, live_miss, live_conf, np.eye(4))

    assert stats["ran"] is True
    assert len(f.announcements) == 1
    header, _ = f.announcements[0]
    assert header["found"] is True
    assert header["basis"] == "memory", "this coordinate came from what t1 saw"
    assert header["x"] == pytest.approx(3.5 + 1.0, abs=0.05), \
        "the newest keyframe was taken at x=3.5, with the object 1 m ahead"


def test_a_live_sighting_supersedes_the_remembered_one():
    """A remembered coordinate is a claim about the past; the present beats it."""
    proc = a_seeking_proc(compare={"camera_z": 0.45, "camera_x": 0.0},
                          seek={"detect_every_n": 1, "min_found_interval_ms": 0.0})
    f = a_t2_frame()
    proc._decide(f, *a_pointmap(), np.eye(4))
    header, _ = f.announcements[0]
    assert header["basis"] == "live"


def test_the_stage_needs_a_pose_to_produce_a_map_frame_coordinate():
    """Without one the cloud sits in CUT3R's own world frame, whose origin
    nobody chose and which no behaviour tree can navigate in."""
    proc = a_seeking_proc()
    f = a_t2_frame()
    f.odom = None
    stats = proc._decide(f, *a_pointmap(), np.eye(4))
    assert stats["ran"] is False
    assert "pose" in stats["why"]
    assert f.announcements == []


def test_a_detector_that_raises_does_not_kill_the_session():
    class Exploding(StubDetector):
        def detect(self, image, queries):
            raise RuntimeError("cuda oom")

    proc = a_seeking_proc(Exploding(), seek={"detect_every_n": 1})
    f = a_t2_frame()
    stats = proc._decide(f, *a_pointmap(), np.eye(4))
    assert stats["ran"] is True
    assert "detect_error" in stats
    assert f.announcements == []
