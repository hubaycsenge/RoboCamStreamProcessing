import numpy as np

from robocam.pipeline import FrameQueue
from robocam.processors.base import Frame
from robocam.processors.stats import StatsProcessor


def make_frame(seq: int, w: int = 32, h: int = 24) -> Frame:
    return Frame(seq=seq, session_id="s", image=np.zeros((h, w, 3), np.uint8), payload_bytes=100)


def test_queue_accepts_up_to_depth():
    q = FrameQueue(max_depth=2)
    assert q.put(make_frame(0)) == []
    assert q.put(make_frame(1)) == []
    assert len(q) == 2


def test_queue_evicts_oldest_when_full():
    q = FrameQueue(max_depth=2, drop_policy="oldest")
    q.put(make_frame(0))
    q.put(make_frame(1))
    evicted = q.put(make_frame(2))

    assert [f.seq for f in evicted] == [0]
    assert len(q) == 2
    # The freshest frames survived, which is the point.
    assert q.get()[1].seq == 1
    assert q.get()[1].seq == 2


def test_queue_refuses_newest_under_that_policy():
    q = FrameQueue(max_depth=2, drop_policy="newest")
    q.put(make_frame(0))
    q.put(make_frame(1))
    rejected = q.put(make_frame(2))

    assert [f.seq for f in rejected] == [2]
    assert q.get()[1].seq == 0


def test_every_dropped_frame_is_reported():
    """The client sizes its in-flight window from replies, so drops must be visible."""
    q = FrameQueue(max_depth=1)
    reported = []
    for seq in range(10):
        reported.extend(f.seq for f in q.put(make_frame(seq)))
    remaining = [f.seq for f in q.drain()]

    assert sorted(reported + remaining) == list(range(10))
    assert q.dropped_total == 9


def test_queue_get_times_out_when_empty():
    q = FrameQueue(max_depth=2)
    assert q.get(timeout=0.01) is None


def test_stats_processor_reports_geometry():
    proc = StatsProcessor(brightness=True, checksum=True)
    proc.setup()
    frame = Frame(seq=0, session_id="s",
                  image=np.full((480, 640, 3), 200, np.uint8), payload_bytes=5000)

    data = proc.process(frame)

    assert data["received"] is True
    assert data["shape"] == [480, 640, 3]
    assert (data["width"], data["height"], data["channels"]) == (640, 480, 3)
    assert data["dtype"] == "uint8"
    assert data["nbytes"] == 640 * 480 * 3
    assert data["mean"] == 200.0
    # A uniform image has no variation, which is exactly the "blank" signal.
    assert data["looks_blank"] is True
    assert "checksum" in data


def test_stats_processor_counts_frames():
    proc = StatsProcessor()
    proc.setup()
    for seq in range(3):
        data = proc.process(make_frame(seq))
    assert data["frames_seen"] == 3
