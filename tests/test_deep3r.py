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
