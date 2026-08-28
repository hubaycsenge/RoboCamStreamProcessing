"""End-to-end deep3r tests that need a GPU, CUT3R and the weights.

Everything here skips unless all three are present, so the suite still runs in
the default venv on a laptop.  What it covers is the half the unit tests
cannot: that the view dict CUT3R is handed is the one it expects, that the
recurrent state actually carries between calls, and that the numbers coming
back are metric rather than merely well-shaped.

Run it on the cluster, on the venv that has torch:

    srun --jobid=<id> --overlap \\
        ./.venv-cut3r/bin/python -m pytest tests/test_deep3r_gpu.py -q
"""

import os

import numpy as np
import pytest

from robocam.processors import build
from robocam.processors.base import Frame

CUT3R_ROOT = os.path.expanduser(os.environ.get("CUT3R_ROOT", "~/CUT3R"))
WEIGHTS = os.path.expanduser(
    os.environ.get("DEEP3R_WEIGHTS", "~/CUT3R/checkpoints/cut3r_512_dpt_4_64.pth")
)


def _why_skip():
    try:
        import torch
    except ImportError:
        return "torch is not installed (this is the CPU-only venv)"
    if not torch.cuda.is_available():
        return "no CUDA device"
    if not os.path.isdir(CUT3R_ROOT):
        return f"no CUT3R checkout at {CUT3R_ROOT}"
    if not os.path.isfile(WEIGHTS):
        return f"no checkpoint at {WEIGHTS}"
    return None


pytestmark = pytest.mark.skipif(_why_skip() is not None, reason=_why_skip() or "")


@pytest.fixture(scope="module")
def proc():
    """One model for the module: loading it costs ~25 s and 3 GB of VRAM."""
    # reset_on_gap is disabled: these tests share one model (loading it twice
    # would cost another 25 s and 3 GB) and pytest-randomly reorders them, so
    # the sequence numbers they use are not contiguous. Gap policy is covered
    # exhaustively in test_deep3r.py, where no GPU is needed to check it.
    p = build("deep3r", {"cut3r_root": CUT3R_ROOT, "weights": WEIGHTS,
                         "max_points": 2000, "reset_on_gap": 10 ** 9})
    p.setup()
    yield p
    p.close()


def scene(i: int, h: int = 480, w: int = 640) -> np.ndarray:
    """A textured scene that pans, so consecutive frames actually overlap."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = i * 12
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (np.sin((xx + s) / 17.0) * 110 + 128).astype(np.uint8)
    img[..., 1] = (np.sin(yy / 23.0) * 110 + 128).astype(np.uint8)
    img[..., 2] = (np.sin((xx + s + yy) / 31.0) * 110 + 128).astype(np.uint8)
    return img


def test_a_frame_produces_a_cloud(proc):
    out = proc.process(Frame(seq=0, session_id="gpu", image=scene(0)))

    assert out["cloud"]["n_raw"] > 10_000
    assert out["cloud"]["frame"] == "cut3r_world"
    assert len(out["pose_c2w"]) == 4


def test_the_cloud_decodes_to_metric_points(proc):
    """The whole approach rests on the scale being metres, not arbitrary units."""
    import base64

    out = proc.process(Frame(seq=1, session_id="gpu", image=scene(1)))
    cloud = out["cloud"]
    if not cloud["n_points"]:
        pytest.skip("synthetic scene produced no confident points")

    raw = np.frombuffer(base64.b64decode(cloud["xyz_u16"]), "<u2").reshape(-1, 3)
    pts = np.asarray(cloud["origin"]) + cloud["scale"] * raw
    depth = np.linalg.norm(pts, axis=1)

    assert np.isfinite(pts).all()
    # An indoor scene in front of a camera: metres, not millimetres and not
    # some normalised unit. This is the assertion that would catch a silent
    # loss of metric scale after a CUT3R upgrade.
    assert 0.1 < np.median(depth) < 50.0


def test_the_state_carries_between_frames(proc):
    """Without the carried state every frame would be its own map."""
    # Establish the state first: whichever test runs first pays the "first
    # frame" reset, and this one must not depend on being that test.
    proc.process(Frame(seq=10, session_id="gpu", image=scene(0)))
    outs = [proc.process(Frame(seq=11 + i, session_id="gpu", image=scene(i + 1)))
            for i in range(3)]

    assert [o["reset"] for o in outs] == [False, False, False]
    assert len({o["map_id"] for o in outs}) == 1
    counts = [o["frames_in_state"] for o in outs]
    assert counts == sorted(counts) and counts[-1] - counts[0] == 2


def test_a_reset_reinitialises_the_model_state(proc):
    """A reset has to rebuild CUT3R's state, not just renumber the map.

    The CPU tests cover *when* a reset fires; what needs a GPU is that the
    model survives having its recurrent state dropped and rebuilt mid-session,
    which is what every reset_every and every gap does in a long traverse.
    """
    before = proc.process(Frame(seq=100, session_id="gpu", image=scene(0)))
    proc._reset_state()
    after = proc.process(Frame(seq=101, session_id="gpu", image=scene(1)))

    assert after["reset"] is True
    assert after["map_id"] > before["map_id"]
    assert after["frames_in_state"] == 1
    assert after["cloud"]["n_raw"] > 10_000


def test_the_payload_stays_within_the_result_envelope(proc):
    """The cap exists so a cloud cannot outgrow the JSON result it rides in."""
    out = proc.process(Frame(seq=200, session_id="gpu", image=scene(3)))
    cloud = out["cloud"]
    payload = len(cloud["xyz_u16"]) + len(cloud["rgb_u8"])

    assert cloud["n_points"] <= 2000
    assert payload < 200_000
