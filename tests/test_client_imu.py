"""The client's IMU path: OpenCR frame parsing, message extraction, and the feed.

Like the LiDAR's serial parser, the OpenCR parser is talking to a device that is
not here, so it is tested against a stream generated from the frame layout it
claims to implement — the same layout the robot's own ``mecanumbot_io_node``
parses.  That catches a mistake in the framing, the CRC or the offset of the IMU
floats within the payload.  It cannot catch a wrong assumption about the board
itself, which is why ``--imu auto`` prefers the ROS 2 topic and why the server
draws the attitude onto the snapshot.

The feed is tested separately because it makes the one decision that differs
from the scanner's: a burst keeps every sample rather than only the newest, and
what it has to discard it must count.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
import robocam_client  # noqa: E402


# -- OpenCR frames ----------------------------------------------------------


def opencr_packet(imu_values, seq: int = 0, shorts=None, dms=0.0, battery=12.0) -> bytes:
    """One board frame: magic, sequence, 27 int16 + 15 float32, CRC-8/0x07."""
    shorts = list(shorts or [0] * 27)
    floats = [dms, battery] + list(imu_values)
    payload = struct.pack(robocam_client.OPENCR_PAYLOAD_FMT, *shorts, *floats)
    body = robocam_client.OPENCR_MAGIC + bytes([seq]) + payload
    return body + bytes([robocam_client._crc8_ccitt(body)])


def level_imu(**over):
    """The 13 IMU floats for a level board at rest."""
    values = {"wx": 0.0, "wy": 0.0, "wz": 0.0,
              "ax": 0.0, "ay": 0.0, "az": 9.80665,
              "mx": 20.0, "my": -5.0, "mz": 40.0,
              "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
    values.update(over)
    return [values[name] for name in robocam_client.IMU_FIELDS]


class FakeOpenCRPort:
    """Hands out a byte stream in chunks, as a serial port would.

    A real port blocks until more bytes arrive, so ``samples()`` loops on an
    empty read forever by design.  Here the stream is finite, so running out
    stands in for the device going away: it asks the source to stop, which is
    what ends the generator.
    """

    def __init__(self, data: bytes, chunk: int = 512) -> None:
        self.data = data
        self.chunk = chunk
        self.pos = 0
        self.closed = False
        self.owner = None

    def read(self, size: int) -> bytes:
        if self.pos >= len(self.data):
            if self.owner is not None:
                self.owner._stopping = True
            return b""
        end = min(self.pos + min(size, self.chunk), len(self.data))
        out = self.data[self.pos:end]
        self.pos = end
        return out

    def close(self) -> None:
        self.closed = True


def make_source(stream: bytes, monkeypatch) -> robocam_client.SerialOpenCRSource:
    """A SerialOpenCRSource reading ``stream``, with no real device involved."""
    source = robocam_client.SerialOpenCRSource.__new__(robocam_client.SerialOpenCRSource)
    source._ser = FakeOpenCRPort(stream)
    source._buf = bytearray()
    source._stopping = False
    source._bytes = 0
    source._packets = 0
    source._crc_bad = 0
    source._implausible = 0
    source.port = "/dev/opencr"
    source.baud = robocam_client.OPENCR_BAUD
    source.rate_hz = robocam_client.OPENCR_RATE_HZ
    source._health = robocam_client._ImuHealth(
        "test", lambda waited: "", robocam_client.OPENCR_RATE_HZ, every=0.0)
    source._ser.owner = source
    return source


def drain(source, limit: int = 1000):
    """Every sample the source produces before its stream runs out."""
    out = []
    for reading in source.samples():
        out.append(reading)
        if len(out) >= limit:
            break
    return out


def test_packet_layout_matches_the_boards_frame_size():
    """118 bytes: 2 magic + 1 sequence + 27*2 + 15*4 + 1 CRC."""
    assert robocam_client.OPENCR_PAYLOAD_BYTES == 27 * 2 + 15 * 4
    assert robocam_client.OPENCR_PACKET_BYTES == 118
    assert len(opencr_packet(level_imu())) == robocam_client.OPENCR_PACKET_BYTES


def test_the_imu_floats_are_taken_from_the_right_offset(monkeypatch):
    """The failure this catches: reading battery voltage as angular rate."""
    values = level_imu(wx=1.5, az=9.5, qw=0.7071, qz=0.7071)
    source = make_source(opencr_packet(values, dms=1.0, battery=11.7), monkeypatch)
    readings = drain(source)
    assert len(readings) == 1
    assert readings[0].values == pytest.approx(tuple(values), abs=1e-4)


def test_the_channel_order_matches_the_protocols_field_names(monkeypatch):
    """Each channel distinct, so a transposition cannot pass unnoticed."""
    values = [float(i + 1) for i in range(len(robocam_client.IMU_FIELDS))]
    source = make_source(opencr_packet(values), monkeypatch)
    got = drain(source)[0].values
    assert dict(zip(robocam_client.IMU_FIELDS, got)) == pytest.approx(
        dict(zip(robocam_client.IMU_FIELDS, values)), abs=1e-4)


def test_a_run_of_frames_is_reassembled_across_read_boundaries(monkeypatch):
    stream = b"".join(opencr_packet(level_imu(wz=float(i)), seq=i % 256) for i in range(20))
    source = make_source(stream, monkeypatch)
    source._ser.chunk = 37  # deliberately not a packet multiple
    readings = drain(source)
    assert len(readings) == 20
    assert [r.values[2] for r in readings] == pytest.approx(list(range(20)))


def test_leading_garbage_before_the_first_magic_is_skipped(monkeypatch):
    source = make_source(b"\x01\x02\x03" + opencr_packet(level_imu()), monkeypatch)
    assert len(drain(source)) == 1


def test_a_corrupt_frame_is_dropped_and_the_stream_resynchronises(monkeypatch):
    bad = bytearray(opencr_packet(level_imu()))
    bad[-1] ^= 0xFF  # break the CRC
    source = make_source(bytes(bad) + opencr_packet(level_imu(wz=2.0)), monkeypatch)
    readings = drain(source)
    assert len(readings) == 1, "the good frame after the corrupt one must survive"
    assert readings[0].values[2] == pytest.approx(2.0)
    assert source._crc_bad == 1


def test_a_magic_pattern_inside_the_payload_does_not_derail_framing(monkeypatch):
    """0x55AA in the wheel telemetry is data, not a header."""
    shorts = [0] * 27
    shorts[3] = struct.unpack("<h", robocam_client.OPENCR_MAGIC)[0]
    stream = (opencr_packet(level_imu(wz=1.0), shorts=shorts)
              + opencr_packet(level_imu(wz=2.0), shorts=shorts))
    source = make_source(stream, monkeypatch)
    readings = drain(source)
    assert len(readings) == 2
    assert [r.values[2] for r in readings] == pytest.approx([1.0, 2.0])


def test_a_frame_that_passes_crc_but_cannot_be_real_is_rejected(monkeypatch):
    """CRC-8 lets one bad resync in 256 through; the float bounds catch it."""
    source = make_source(opencr_packet(level_imu(wx=5000.0))
                         + opencr_packet(level_imu(wz=3.0)), monkeypatch)
    readings = drain(source)
    assert len(readings) == 1
    assert readings[0].values[2] == pytest.approx(3.0)
    assert source._implausible == 1


def test_the_buffer_stays_bounded_on_a_stream_that_never_frames(monkeypatch):
    source = make_source(b"\x00" * 40000, monkeypatch)
    drain(source, limit=1)
    assert len(source._buf) <= 8192


# -- plausibility -----------------------------------------------------------


def test_non_finite_values_are_implausible():
    assert not robocam_client._plausible_imu(level_imu(az=float("nan")))
    assert not robocam_client._plausible_imu(level_imu(qw=float("inf")))


@pytest.mark.parametrize("over, ok", [
    ({"wx": 99.0}, True), ({"wx": 101.0}, False),
    ({"ax": 199.0}, True), ({"ax": 201.0}, False),
])
def test_the_bounds_are_the_sensors_range_not_a_tolerance(over, ok):
    assert robocam_client._plausible_imu(level_imu(**over)) is ok


def test_an_ordinary_sample_is_plausible():
    assert robocam_client._plausible_imu(level_imu(wz=1.5, ax=0.4))


# -- ROS 2 message extraction -----------------------------------------------


class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class _FakeOpenCRState:
    """Only the IMU fields the extractor reads, named as the real message names them."""

    def __init__(self):
        for i, name in enumerate((
            "imu_angular_vel_x", "imu_angular_vel_y", "imu_angular_vel_z",
            "imu_linear_acc_x", "imu_linear_acc_y", "imu_linear_acc_z",
            "imu_magnetic_x", "imu_magnetic_y", "imu_magnetic_z",
            "imu_orientation_w", "imu_orientation_x",
            "imu_orientation_y", "imu_orientation_z",
        )):
            setattr(self, name, float(i + 1))
        self.battery_voltage = 11.7  # must not appear in the output


def test_opencr_state_extraction_keeps_the_field_order():
    got = robocam_client._from_opencr_state(_FakeOpenCRState())
    assert got == pytest.approx(tuple(float(i + 1) for i in range(13)))
    assert len(got) == len(robocam_client.IMU_FIELDS)
    assert 11.7 not in got


def test_sensor_msgs_imu_extraction_has_no_magnetometer():
    msg = type("Imu", (), {})()
    msg.angular_velocity = _Vec(1.0, 2.0, 3.0)
    msg.linear_acceleration = _Vec(4.0, 5.0, 6.0)
    msg.orientation = _Vec(x=8.0, y=9.0, z=10.0, w=7.0)
    got = robocam_client._from_ros_imu(msg)
    assert got == pytest.approx((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0))
    assert len(got) == len(robocam_client.IMU_FIELDS_NO_MAG)
    assert "mx" not in robocam_client.IMU_FIELDS_NO_MAG


# -- the quaternion the synthetic source builds -----------------------------


@pytest.mark.parametrize("roll, pitch, yaw", [
    (0.0, 0.0, 0.0), (25.0, 0.0, 0.0), (0.0, -15.0, 0.0), (10.0, 20.0, 30.0),
])
def test_the_clients_quaternion_matches_the_servers_decode(roll, pitch, yaw):
    """Both ends must agree on the convention, or the fake data proves nothing."""
    from robocam import imu as server_imu

    q = robocam_client._euler_to_quat(*(math.radians(v) for v in (roll, pitch, yaw)))
    assert server_imu.quat_to_euler_deg(q) == pytest.approx((roll, pitch, yaw), abs=1e-4)


def test_the_synthetic_sources_accelerometer_agrees_with_its_own_quaternion():
    """The point of the synthetic board: if the server's two attitude routes
    disagree, that is a server bug rather than a quirk of the fake data."""
    from robocam import imu as server_imu

    source = robocam_client.SyntheticImuSource(hz=1000.0, tilt_deg=6.0)
    samples = []
    for reading in source.samples():
        samples.append(reading)
        if len(samples) >= 5:
            source.request_stop()
            break

    for reading in samples:
        _, _, _, ax, ay, az, _, _, _, qw, qx, qy, qz = reading.values
        roll_q, pitch_q, _ = server_imu.quat_to_euler_deg((qw, qx, qy, qz))
        roll_a = math.degrees(math.atan2(ay, az))
        pitch_a = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
        assert roll_a == pytest.approx(roll_q, abs=0.05)
        assert pitch_a == pytest.approx(pitch_q, abs=0.05)


def test_synthetic_imu_says_it_is_not_a_real_sensor():
    info = robocam_client.SyntheticImuSource().info()
    assert info["model"] == "synthetic"
    assert "SYNTHETIC" in info["note"]


def test_synthetic_imu_is_never_chosen_automatically(monkeypatch):
    """``--imu auto`` must fail honestly rather than invent inertial data."""
    def refuse(*args, **kwargs):
        raise robocam_client.ImuUnavailable("nothing here")

    monkeypatch.setattr(robocam_client, "Ros2OpenCRSource", refuse)
    monkeypatch.setattr(robocam_client, "SerialOpenCRSource", refuse)

    args = type("Args", (), {
        "imu": "auto", "imu_topic": "/opencr_state", "imu_msg": "auto",
        "imu_port": "/dev/opencr", "imu_baud": robocam_client.OPENCR_BAUD,
        "imu_allow_shared_port": False, "imu_health_every": 0.0, "imu_hz": 100.0,
    })()
    assert robocam_client.build_imu_source(args) is None


# -- the feed ---------------------------------------------------------------


class ListImuSource(robocam_client.ImuSource):
    """Yields a fixed list of samples, then stops."""

    model = "list"

    def __init__(self, count: int) -> None:
        self.rate_hz = 100.0
        self.count = count
        self.closed = False

    def samples(self):
        for i in range(self.count):
            yield robocam_client.ImuReading((float(i),) * len(robocam_client.IMU_FIELDS))

    def close(self) -> None:
        self.closed = True


def _run_feed(feed) -> None:
    """Run the reader synchronously, so the test does not race a thread."""
    feed._run()


def test_the_feed_keeps_every_sample_not_just_the_newest():
    """The difference from the scanner: a dropped shake is gone for good."""
    feed = robocam_client.ImuFeed(ListImuSource(50), max_samples=400)
    _run_feed(feed)
    batch, dropped = feed.take()
    assert len(batch) == 50
    assert dropped == 0
    assert [r.values[0] for r in batch] == pytest.approx([float(i) for i in range(50)])


def test_taking_twice_does_not_resend_the_same_samples():
    feed = robocam_client.ImuFeed(ListImuSource(10))
    _run_feed(feed)
    assert len(feed.take()[0]) == 10
    assert feed.take()[0] == []


def test_overflow_drops_the_oldest_and_counts_what_it_dropped():
    """An unbounded queue on a robot is a memory leak with extra steps."""
    feed = robocam_client.ImuFeed(ListImuSource(30), max_samples=10)
    _run_feed(feed)
    batch, dropped = feed.take()
    assert len(batch) == 10
    assert dropped == 20
    # The newest survive: the oldest are the ones worth losing.
    assert [r.values[0] for r in batch] == pytest.approx([float(i) for i in range(20, 30)])


def test_the_dropped_count_resets_once_reported():
    """It travels with one burst; counting it twice would overstate the gap."""
    feed = robocam_client.ImuFeed(ListImuSource(30), max_samples=10)
    _run_feed(feed)
    assert feed.take()[1] == 20
    assert feed.take()[1] == 0


def test_a_board_that_fails_does_not_take_the_camera_stream_with_it():
    class Exploding(robocam_client.ImuSource):
        def samples(self):
            raise OSError("board unplugged")
            yield  # pragma: no cover

    feed = robocam_client.ImuFeed(Exploding())
    _run_feed(feed)
    assert feed.failed is True


def test_a_shutdown_race_is_not_reported_as_a_failure():
    """The device closed under a reader that was already asked to stop."""
    class Exploding(robocam_client.ImuSource):
        def samples(self):
            raise OSError("closed during shutdown")
            yield  # pragma: no cover

    feed = robocam_client.ImuFeed(Exploding())
    feed._stop.set()
    _run_feed(feed)
    assert feed.failed is False


def test_the_feed_closes_the_device_when_it_stops():
    source = ListImuSource(5)
    feed = robocam_client.ImuFeed(source)
    feed.start()
    feed.stop(timeout=2.0)
    assert source.closed is True
