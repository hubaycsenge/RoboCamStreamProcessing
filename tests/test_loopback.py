"""End-to-end tests: a real server on a real socket, a real client.

These are the tests that would have caught a wire-format mistake, so they use
TCP on localhost rather than mocking the transport.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import zmq

from robocam import lidar, wire
from robocam.config import Config
from robocam.server import StreamServer

# The client is deployed to the robot as a standalone file, so it is not on the
# package path; add its directory explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
import robocam_client  # noqa: E402


@contextlib.contextmanager
def running_server(**overrides):
    """A server on an ephemeral localhost port, with config overrides applied."""
    raw = {
        "server": {"bind": "tcp://127.0.0.1:0", "session_timeout_s": 0},
        "processor": {"name": "stats", "options": {"brightness": True}},
        "snapshot": {"enabled": False},
        "logging": {"stats_interval_s": 0},
    }
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)

    srv = StreamServer(Config.from_dict(raw))
    srv.start()
    endpoint = srv.sock.getsockopt(zmq.LAST_ENDPOINT).decode()

    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    try:
        yield srv, endpoint
    finally:
        srv.stop()
        thread.join(timeout=5)


@pytest.fixture
def server():
    """A running server bound to an ephemeral localhost port."""
    with running_server() as running:
        yield running


def jpeg(width: int, height: int) -> bytes:
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def room_scan(count: int = 360, wall_m: float = 3.0, obstacle_deg=None, obstacle_m: float = 0.4):
    """Ranges for a circular wall, optionally with something closer in one direction."""
    ranges = np.full(count, wall_m, dtype=np.float32)
    if obstacle_deg is not None:
        ranges[int(round(obstacle_deg % 360 / 360.0 * count)) % count] = obstacle_m
    return ranges


class RawPeer:
    """A bare DEALER socket, to exercise the protocol without the client."""

    def __init__(self, endpoint: str, identity: bytes = b"test-peer"):
        self.sock = zmq.Context.instance().socket(zmq.DEALER)
        self.sock.setsockopt(zmq.IDENTITY, identity)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(endpoint)

    def send(self, header, payload=b""):
        head, body = wire.encode(header, payload)
        self.sock.send_multipart([head, body])

    def send_scan(self, seq: int, ranges_m, **overrides):
        header = wire.scan(seq=seq, count=len(ranges_m), t_capture_ns=seq,
                           range_min=0.12, range_max=12.0, source="pytest")
        header.update(overrides)
        self.send(header, lidar.encode_scan_payload(ranges_m))

    def recv(self, timeout_ms: int = 3000):
        if not self.sock.poll(timeout_ms):
            raise TimeoutError("no reply from server")
        parts = self.sock.recv_multipart()
        return json.loads(parts[0].decode())

    def recv_of_type(self, mtype: str, timeout_ms: int = 3000):
        """Next message of a given type, skipping the other stream's replies."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            message = self.recv(timeout_ms=timeout_ms)
            if message.get("type") == mtype:
                return message
        raise TimeoutError(f"no {mtype} from server")

    def close(self):
        self.sock.close(linger=0)


def test_handshake(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"handshake-peer")
    try:
        peer.send(wire.hello("robot-1", codec=wire.CODEC_JPEG, width=640, height=480, fps=30))
        reply = peer.recv()

        assert reply["type"] == wire.MSG_WELCOME
        assert reply["accepted"] is True
        assert reply["processor"] == "stats"
        assert "host" in reply["server"]
    finally:
        peer.close()


def test_frame_result_reports_actual_geometry(server):
    """The result must describe what the server decoded, not what the client claimed."""
    _, endpoint = server
    peer = RawPeer(endpoint, b"geometry-peer")
    try:
        peer.send(wire.hello("robot-1"))
        assert peer.recv()["type"] == wire.MSG_WELCOME

        payload = jpeg(640, 480)
        # Deliberately lie about the dimensions in the header.
        peer.send(wire.frame(seq=1, codec=wire.CODEC_JPEG, width=99, height=99,
                             t_capture_ns=12345), payload)
        result = peer.recv()

        assert result["type"] == wire.MSG_RESULT
        assert result["seq"] == 1
        assert result["ok"] is True
        assert result["reason"] == "ok"
        assert (result["width"], result["height"], result["channels"]) == (640, 480, 3)
        assert result["dtype"] == "uint8"
        assert result["nbytes"] == 640 * 480 * 3
        assert result["payload_bytes"] == len(payload)
        assert result["data"]["received"] is True
        assert result["data"]["shape"] == [480, 640, 3]
        # Client timestamps come back untouched for RTT computation.
        assert result["t_capture_ns"] == 12345
    finally:
        peer.close()


def test_raw_bgr_codec(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"raw-peer")
    try:
        peer.send(wire.hello("robot-1", codec=wire.CODEC_RAW_BGR))
        assert peer.recv()["accepted"] is True

        img = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        peer.send(wire.frame(seq=0, codec=wire.CODEC_RAW_BGR, width=160, height=120,
                             t_capture_ns=1), img.tobytes())
        result = peer.recv()

        assert result["ok"] is True
        assert (result["width"], result["height"]) == (160, 120)
    finally:
        peer.close()


def test_corrupt_payload_reports_failure_without_killing_the_session(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"corrupt-peer")
    try:
        peer.send(wire.hello("robot-1"))
        peer.recv()

        peer.send(wire.frame(seq=1, codec=wire.CODEC_JPEG, width=64, height=64,
                             t_capture_ns=1), b"this is not a jpeg")
        bad = peer.recv()
        assert bad["ok"] is False
        assert bad["reason"] == wire.REASON_DECODE_FAILED

        # The session must survive so a glitch does not drop the link.
        peer.send(wire.frame(seq=2, codec=wire.CODEC_JPEG, width=64, height=64,
                             t_capture_ns=2), jpeg(64, 64))
        good = peer.recv()
        assert good["ok"] is True
        assert good["seq"] == 2
    finally:
        peer.close()


def test_unsupported_codec_rejected_at_handshake(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"badcodec-peer")
    try:
        peer.send(wire.hello("robot-1", codec="vp9"))
        reply = peer.recv()
        assert reply["accepted"] is False
        assert "unsupported codec" in reply["message"]
    finally:
        peer.close()


def test_protocol_version_mismatch_rejected(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"oldproto-peer")
    try:
        header = wire.hello("robot-1")
        header["protocol"] = 999
        peer.send(header)
        reply = peer.recv()
        assert reply["accepted"] is False
        assert "protocol" in reply["message"]
    finally:
        peer.close()


def test_ping_pong(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"ping-peer")
    try:
        peer.send(wire.hello("robot-1"))
        peer.recv()
        peer.send(wire.ping(42))
        pong = peer.recv()
        assert pong["type"] == wire.MSG_PONG
        assert pong["nonce"] == 42
    finally:
        peer.close()


def test_every_frame_gets_exactly_one_result(server):
    """Including frames the queue drops — the client's flow control depends on it."""
    srv, endpoint = server
    peer = RawPeer(endpoint, b"count-peer")
    try:
        peer.send(wire.hello("robot-1"))
        peer.recv()

        n = 40
        payload = jpeg(320, 240)
        for seq in range(n):
            peer.send(wire.frame(seq=seq, codec=wire.CODEC_JPEG, width=320, height=240,
                                 t_capture_ns=seq), payload)

        seen = set()
        deadline = time.monotonic() + 10
        while len(seen) < n and time.monotonic() < deadline:
            try:
                msg = peer.recv(timeout_ms=1000)
            except TimeoutError:
                break
            if msg.get("type") == wire.MSG_RESULT:
                assert msg["seq"] not in seen, "duplicate result for a frame"
                seen.add(msg["seq"])

        assert seen == set(range(n)), f"missing results for {sorted(set(range(n)) - seen)}"
    finally:
        peer.close()


# ---------------------------------------------------------------------------
# LiDAR
# ---------------------------------------------------------------------------


def test_scan_gets_a_result_with_the_obstacle_summary(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"scan-peer")
    try:
        peer.send(wire.hello("robot-1", lidar={"model": "lds02", "points": 360}))
        assert peer.recv()["accepted"] is True

        peer.send_scan(0, room_scan(obstacle_deg=0, obstacle_m=0.3))
        result = peer.recv_of_type(wire.MSG_SCAN_RESULT)

        assert result["seq"] == 0
        assert result["ok"] is True
        assert result["points"] == 360
        data = result["data"]
        assert data["nearest_m"] == pytest.approx(0.3, abs=0.01)
        assert data["nearest_deg"] == pytest.approx(0.0, abs=1.0)
        assert data["obstacle"] is True
        assert data["coverage"] == 1.0
        assert len(data["sector_min_m"]) == 12
        # Timestamps come back untouched, as for frames.
        assert result["t_capture_ns"] == 0
    finally:
        peer.close()


def test_welcome_advertises_the_servers_lidar_geometry(server):
    """The client needs to know whether to spin the scanner up at all."""
    _, endpoint = server
    peer = RawPeer(endpoint, b"advert-peer")
    try:
        peer.send(wire.hello("robot-1"))
        info = peer.recv()["server"]["lidar"]
        assert info["enabled"] is True
        assert "u16mm" in info["encodings"]
        assert "mount_yaw_deg" in info and "camera_hfov_deg" in info
    finally:
        peer.close()


def test_scan_is_attached_to_the_next_frame(server):
    """The pairing that makes the two sensors worth having on one link."""
    _, endpoint = server
    peer = RawPeer(endpoint, b"pair-peer")
    try:
        peer.send(wire.hello("robot-1"))
        peer.recv()

        peer.send_scan(7, room_scan(obstacle_deg=90, obstacle_m=0.6))
        assert peer.recv_of_type(wire.MSG_SCAN_RESULT)["ok"] is True

        peer.send(wire.frame(seq=1, codec=wire.CODEC_JPEG, width=320, height=240,
                             t_capture_ns=1), jpeg(320, 240))
        result = peer.recv_of_type(wire.MSG_RESULT)

        assert result["scan_seq"] == 7
        assert result["scan_age_ms"] >= 0
        summary = result["data"]["lidar"]
        assert summary is not None
        assert summary["nearest_m"] == pytest.approx(0.6, abs=0.01)
        assert summary["nearest_deg"] == pytest.approx(90.0, abs=1.0)
        assert result["data"]["scan_fraction"] == 1.0
    finally:
        peer.close()


def test_a_frame_with_no_scan_says_so_rather_than_omitting_it(server):
    """A robot that reads a missing key as "clear ahead" is a robot that crashes."""
    _, endpoint = server
    peer = RawPeer(endpoint, b"noscan-peer")
    try:
        peer.send(wire.hello("robot-1"))
        peer.recv()
        peer.send(wire.frame(seq=1, codec=wire.CODEC_JPEG, width=320, height=240,
                             t_capture_ns=1), jpeg(320, 240))
        result = peer.recv_of_type(wire.MSG_RESULT)

        assert result["ok"] is True
        assert "scan_seq" not in result
        assert result["data"]["lidar"] is None
    finally:
        peer.close()


def test_a_stale_scan_is_not_fused_into_a_frame():
    """Old ranges describe an old world; the frame must arrive without them."""
    with running_server(lidar={"stale_after_ms": 50.0}) as (_, endpoint):
        peer = RawPeer(endpoint, b"stale-peer")
        try:
            peer.send(wire.hello("robot-1"))
            peer.recv()
            peer.send_scan(0, room_scan())
            assert peer.recv_of_type(wire.MSG_SCAN_RESULT)["ok"] is True

            time.sleep(0.15)
            peer.send(wire.frame(seq=1, codec=wire.CODEC_JPEG, width=160, height=120,
                                 t_capture_ns=1), jpeg(160, 120))
            result = peer.recv_of_type(wire.MSG_RESULT)

            assert "scan_seq" not in result
            assert result["data"]["lidar"] is None
        finally:
            peer.close()


def test_a_malformed_scan_does_not_kill_the_session(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"badscan-peer")
    try:
        peer.send(wire.hello("robot-1"))
        peer.recv()

        header = wire.scan(seq=1, count=360, t_capture_ns=0)
        peer.send(header, b"\x00\x01\x02")          # far too short for 360 points
        bad = peer.recv_of_type(wire.MSG_SCAN_RESULT)
        assert bad["ok"] is False
        assert bad["reason"] == wire.REASON_BAD_SCAN
        assert "expected at least" in bad["data"]["error"]

        peer.send_scan(2, room_scan())
        good = peer.recv_of_type(wire.MSG_SCAN_RESULT)
        assert good["ok"] is True
        assert good["seq"] == 2
    finally:
        peer.close()


def test_a_server_with_lidar_disabled_refuses_scans_but_keeps_answering():
    with running_server(lidar={"enabled": False}) as (_, endpoint):
        peer = RawPeer(endpoint, b"nolidar-peer")
        try:
            peer.send(wire.hello("robot-1"))
            assert peer.recv()["server"]["lidar"]["enabled"] is False

            peer.send_scan(0, room_scan())
            refused = peer.recv_of_type(wire.MSG_SCAN_RESULT)
            assert refused["ok"] is False
            assert refused["reason"] == wire.REASON_LIDAR_DISABLED

            # Every scan still gets exactly one reply, so the client's flow
            # control stays healthy while it works out what to do.
            peer.send(wire.frame(seq=1, codec=wire.CODEC_JPEG, width=160, height=120,
                                 t_capture_ns=1), jpeg(160, 120))
            assert peer.recv_of_type(wire.MSG_RESULT)["ok"] is True
        finally:
            peer.close()


def test_fusion_processor_turns_bearings_into_image_columns():
    """The whole point of the pairing: a column of the image, in metres."""
    with running_server(processor={"name": "fusion"},
                        lidar={"camera_hfov_deg": 70.0, "fov_bins": 8}) as (_, endpoint):
        peer = RawPeer(endpoint, b"fusion-peer")
        try:
            peer.send(wire.hello("robot-1"))
            assert peer.recv()["processor"] == "fusion"

            # One return dead ahead, one 20 deg counter-clockwise (to the left).
            ranges = np.full(360, np.nan, dtype=np.float32)
            ranges[0] = 2.0
            ranges[20] = 1.0
            peer.send_scan(0, ranges)
            assert peer.recv_of_type(wire.MSG_SCAN_RESULT)["ok"] is True

            peer.send(wire.frame(seq=1, codec=wire.CODEC_JPEG, width=640, height=480,
                                 t_capture_ns=1), jpeg(640, 480))
            data = peer.recv_of_type(wire.MSG_RESULT)["data"]

            bins = data["fov_bins_m"]
            assert len(bins) == 8
            assert bins[4] == pytest.approx(2.0, abs=0.01), "forward return belongs mid-frame"
            # The nearer return is to the robot's left, which is the left of the
            # image: low columns, low bin indices.
            assert data["nearest_in_view_m"] == pytest.approx(1.0, abs=0.01)
            assert data["nearest_in_view_px"] < 320
            assert data["lidar"]["nearest_deg"] == pytest.approx(20.0, abs=1.0)
        finally:
            peer.close()


def test_snapshot_with_a_scan_is_written_and_annotated(tmp_path):
    """The overlay is a debugging tool, so it must survive contact with real data."""
    with running_server(
        snapshot={"enabled": True, "dir": str(tmp_path), "every_n_frames": 1,
                  "lidar_overlay": True},
    ) as (_, endpoint):
        peer = RawPeer(endpoint, b"snap-peer")
        try:
            peer.send(wire.hello("robot-1"))
            peer.recv()
            peer.send_scan(0, room_scan(obstacle_deg=15, obstacle_m=0.5))
            peer.recv_of_type(wire.MSG_SCAN_RESULT)
            peer.send(wire.frame(seq=1, codec=wire.CODEC_JPEG, width=640, height=480,
                                 t_capture_ns=1), jpeg(640, 480))
            peer.recv_of_type(wire.MSG_RESULT)

            target = tmp_path / "latest.jpg"
            deadline = time.monotonic() + 5
            while not target.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert target.exists(), "no snapshot written"

            written = cv2.imread(str(target))
            assert written is not None and written.shape[:2] == (480, 640)
        finally:
            peer.close()


def test_real_client_streams_camera_and_lidar_together(server):
    """The deployable client, both sensors, against a real server."""
    _, endpoint = server
    results, scan_results = [], []

    client = robocam_client.RoboCamClient(
        server=endpoint,
        client_id="pytest-orin-lidar",
        jpeg_quality=70,
        max_inflight=2,
        on_result=results.append,
        on_scan_result=scan_results.append,
    )
    client.run(
        robocam_client.SyntheticSource(320, 240, fps=30),
        duration=2.0,
        status_every=0,
        scan_source=robocam_client.SyntheticScanSource(points=360, hz=10.0),
    )

    ok_scans = [r for r in scan_results if r.get("ok")]
    assert ok_scans, f"no scan results (got {len(scan_results)} messages)"
    # The synthetic room's nearest wall is 2 m and its obstacle 0.9 m; anything
    # outside that band means the ranges were mangled somewhere on the way.
    assert 0.5 < ok_scans[-1]["data"]["nearest_m"] < 2.5
    assert ok_scans[-1]["data"]["coverage"] > 0.9
    assert ok_scans[-1]["rtt_ms"] > 0

    fused = [r for r in results if r.get("ok") and r.get("scan_seq") is not None]
    assert fused, "no frame was paired with a scan"
    assert fused[-1]["data"]["lidar"]["nearest_m"] > 0
    assert fused[-1]["scan_age_ms"] < 400


def test_real_client_against_real_server(server):
    """The deployable client file, talking to the server, with synthetic frames."""
    _, endpoint = server
    results = []

    client = robocam_client.RoboCamClient(
        server=endpoint,
        client_id="pytest-orin",
        jpeg_quality=70,
        max_inflight=2,
        on_result=results.append,
    )
    source = robocam_client.SyntheticSource(320, 240, fps=30)
    client.run(source, duration=2.0, status_every=0)

    ok = [r for r in results if r.get("ok")]
    assert ok, f"no successful results (got {len(results)} messages)"

    first = ok[0]
    assert first["width"] == 320
    assert first["height"] == 240
    assert first["channels"] == 3
    assert first["data"]["received"] is True
    # The synthetic pattern is not blank, and RTT was measured client-side.
    assert first["data"]["looks_blank"] is False
    assert first["rtt_ms"] > 0
    # JPEG should be well under the raw size.
    assert first["payload_bytes"] < first["nbytes"]
