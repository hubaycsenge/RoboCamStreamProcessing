"""The client's LiDAR path: packet parsing, packing, and the fake room.

The serial parser is the part of this project that cannot be checked against the
real thing from here — it is talking to a device on the robot — so it is tested
against a stream generated from the LD08/LD19 packet layout it claims to
implement.  That catches a mistake in the framing, the angle interpolation or
the binning.  It cannot catch a wrong assumption about the device itself, which
is why the client has --lidar-spin and --lidar-crc as escape hatches and why the
server draws the scan onto the snapshot.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "link"))
import robocam_client  # noqa: E402


# -- CRC --------------------------------------------------------------------


def _crc8_bitwise(data: bytes, poly: int = 0x4D) -> int:
    """Reference CRC-8, computed a bit at a time rather than from the table."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def test_crc_table_matches_a_bitwise_implementation():
    """Validates the table, not the polynomial.

    The polynomial (0x4D) comes from the LD08/LD19 packet spec and can only be
    confirmed against hardware; ``--lidar-crc auto`` is what copes if a
    particular unit disagrees.
    """
    for sample in (b"", b"\x54", b"\x54\x2c\x00\x01", bytes(range(46))):
        assert robocam_client._crc8(sample) == _crc8_bitwise(sample)


# -- packing ----------------------------------------------------------------


def test_pack_mm_treats_inf_nan_and_zero_as_no_return():
    packed = robocam_client._pack_mm([1.5, float("inf"), float("nan"), 0.0, -1.0])
    assert list(packed) == [1500, 0, 0, 0, 0]


def test_pack_mm_clips_rather_than_wrapping():
    """A 70 m reading must not come out as a few centimetres."""
    packed = robocam_client._pack_mm([100.0])
    assert packed[0] == 65535


# -- the LD08 serial parser -------------------------------------------------


def ld08_packet(start_cdeg: int, distances_mm, intensity: int = 200) -> bytes:
    """One 47-byte LD08/LD19 packet spanning 12 points at 1 degree each."""
    assert len(distances_mm) == 12
    end_cdeg = (start_cdeg + 1100) % 36000
    body = struct.pack("<BBHH", 0x54, 0x2C, 2000, start_cdeg)
    for distance in distances_mm:
        body += struct.pack("<HB", distance, intensity)
    body += struct.pack("<HH", end_cdeg, 1234)
    return body + bytes([robocam_client._crc8(body)])


def revolution_bytes() -> bytes:
    """A full turn where the point at N degrees reads (1000 + N) mm."""
    out = b""
    for packet_index in range(30):
        start = packet_index * 1200
        distances = [1000 + packet_index * 12 + i for i in range(12)]
        out += ld08_packet(start, distances)
    return out


class FakeSerialPort:
    """Replays a byte stream in chunks, then loops — like a spinning scanner."""

    def __init__(self, data: bytes, chunk: int = 137) -> None:
        self.data = data
        self.chunk = chunk
        self.offset = 0

    def read(self, _size: int) -> bytes:
        out = self.data[self.offset:self.offset + self.chunk]
        self.offset = (self.offset + self.chunk) % len(self.data)
        return out

    def close(self) -> None:
        pass


@pytest.fixture
def fake_serial(monkeypatch):
    """Stand in for pyserial so the real constructor path is exercised."""
    import types

    def install(data: bytes):
        module = types.ModuleType("serial")
        module.Serial = lambda *a, **kw: FakeSerialPort(data)
        monkeypatch.setitem(sys.modules, "serial", module)

    return install


def test_serial_source_reassembles_a_revolution(fake_serial):
    fake_serial(revolution_bytes())
    source = robocam_client.SerialLdsSource(port="/dev/fake", points=360)

    reading = next(source.scans())

    assert len(reading) == 360
    # Every bin filled, and each one carrying the distance for its own angle:
    # this is the assertion that fails if the angle interpolation is wrong.
    expected = np.arange(360, dtype=np.uint16) + 1000
    np.testing.assert_array_equal(reading.ranges_mm, expected)
    assert reading.intensities is not None
    assert int(reading.intensities[0]) == 200


def test_serial_source_declares_clockwise_spin_in_the_increment_sign(fake_serial):
    """The device turns clockwise; the protocol is counter-clockwise-positive."""
    fake_serial(revolution_bytes())

    cw = next(robocam_client.SerialLdsSource(port="/dev/fake", spin="cw").scans())
    ccw = next(robocam_client.SerialLdsSource(port="/dev/fake", spin="ccw").scans())

    assert cw.angle_increment < 0
    assert ccw.angle_increment > 0
    assert cw.angle_increment == pytest.approx(-ccw.angle_increment)


def test_serial_source_survives_a_corrupt_packet(fake_serial):
    """A bad CRC must cost one packet's worth of points, not the connection."""
    stream = bytearray(revolution_bytes())
    stream[10] ^= 0xFF                      # corrupt a distance mid-packet
    fake_serial(bytes(stream))

    reading = next(robocam_client.SerialLdsSource(port="/dev/fake", crc="check").scans())

    filled = int((reading.ranges_mm > 0).sum())
    assert filled >= 300, "one bad packet should not empty the revolution"
    assert filled < 360, "the corrupt packet's points should have been rejected"


def test_serial_source_ignores_a_header_pattern_inside_the_data(fake_serial):
    """0x54 0x2C occurs in distance fields; resyncing must not eat real packets."""
    distances = [0x2C54, 0x2C54] + [1500] * 10
    stream = ld08_packet(0, distances) + revolution_bytes()
    fake_serial(stream)

    reading = next(robocam_client.SerialLdsSource(port="/dev/fake").scans())
    assert int((reading.ranges_mm > 0).sum()) >= 300


# -- diagnosing a stream that never frames ----------------------------------
#
# The case that cost an afternoon: right port, right device, right packet
# format, wrong baud.  The LDS-02's LD08 runs at 115200 while the LD06 and LD19
# it shares a packet layout with run at 230400, so a reader left at 230400
# samples every bit twice — bytes arrive steadily, at roughly double the true
# rate, and not one of them frames.  "Bytes but no packets" alone cannot say
# which way to move; the byte *rate* can, and that is what these pin.


def _no_frame_message(fake_serial, baud: int, byte_rate: float, waited_s: float = 5.0) -> str:
    fake_serial(b"\x00" * 64)
    source = robocam_client.SerialLdsSource(port="/dev/fake", baud=baud)
    source._bytes = int(byte_rate * waited_s)
    return source._diagnose(waited_s)


def test_default_baud_is_the_lds02_rate():
    """The one number that decides whether the serial path works at all."""
    assert robocam_client.LDS02_BAUD == 115200
    import inspect

    signature = inspect.signature(robocam_client.SerialLdsSource.__init__)
    assert signature.parameters["baud"].default == robocam_client.LDS02_BAUD


def test_too_many_bytes_names_the_lower_baud_to_try(fake_serial):
    """Reading at 230400 what is sent at 115200: ~2x the bytes, none of them framing."""
    message = _no_frame_message(fake_serial, baud=230400, byte_rate=13414.0)

    assert "115200" in message
    assert "--lidar-baud 115200" in message
    assert "1.9x" in message


def test_too_few_bytes_names_the_higher_baud_to_try(fake_serial):
    message = _no_frame_message(fake_serial, baud=115200, byte_rate=1500.0)

    assert "--lidar-baud 230400" in message


def test_a_plausible_rate_that_never_frames_blames_the_device_not_the_baud(fake_serial):
    """Right rate, wrong bytes — the baud is fine and something else is on the port."""
    message = _no_frame_message(
        fake_serial, baud=robocam_client.LDS02_BAUD,
        byte_rate=robocam_client.SerialLdsSource.EXPECTED_BYTES_PER_S,
    )

    assert "--lidar-baud" not in message
    assert "by-id" in message


def test_expected_byte_rate_matches_the_packet_layout():
    """Guards the constant against a change to the packet size or point count."""
    # 360 points a revolution, 12 to a 47-byte packet, 5 revolutions a second.
    assert robocam_client.SerialLdsSource.EXPECTED_BYTES_PER_S == pytest.approx(7050.0)


# -- shutdown ---------------------------------------------------------------


def test_feed_stops_the_reader_before_closing_the_device(fake_serial):
    """Closing under a blocked read is what made every Ctrl-C print a traceback."""
    fake_serial(revolution_bytes())
    source = robocam_client.SerialLdsSource(port="/dev/fake")
    order = []
    source._ser.close = lambda: order.append("close")
    original = source.request_stop
    source.request_stop = lambda: (order.append("request_stop"), original())

    feed = robocam_client.LidarFeed(source)
    feed.start()
    feed.stop(timeout=5.0)

    assert order == ["request_stop", "close"]
    assert not feed.failed, "a clean shutdown must not mark the feed as failed"


def test_feed_does_not_report_a_shutdown_error_as_a_failure(fake_serial, caplog):
    """A source that raises on the way out during stop() is noise, not a fault."""
    fake_serial(revolution_bytes())
    source = robocam_client.SerialLdsSource(port="/dev/fake")

    def explode(_size):
        raise TypeError("'NoneType' object cannot be interpreted as an integer")

    feed = robocam_client.LidarFeed(source)
    feed._stop.set()                    # as stop() would have done
    source._ser.read = explode
    feed._run()                         # synchronous, so the exception lands here

    assert not feed.failed
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_feed_still_reports_a_genuine_failure(fake_serial, caplog):
    """The unplugged-scanner path must keep shouting."""
    fake_serial(revolution_bytes())
    source = robocam_client.SerialLdsSource(port="/dev/fake")
    source._ser.read = lambda _size: (_ for _ in ()).throw(OSError("device disconnected"))

    feed = robocam_client.LidarFeed(source)
    feed._run()

    assert feed.failed
    assert [r for r in caplog.records if r.levelname == "ERROR"]


# -- the synthetic scanner --------------------------------------------------


def test_synthetic_scan_describes_its_room():
    """Walls at the distances the room says, so a sign error shows up as a shape."""
    source = robocam_client.SyntheticScanSource(points=360, hz=1000.0,
                                                room=(-2.0, 4.0, -3.0, 3.0))
    reading = next(source.scans())
    metres = reading.ranges_mm.astype(np.float64) / 1000.0

    assert metres[90] == pytest.approx(3.0, abs=0.01)    # +y wall
    assert metres[180] == pytest.approx(2.0, abs=0.01)   # -x wall
    assert metres[270] == pytest.approx(3.0, abs=0.01)   # -y wall
    # Straight ahead the orbiting obstacle is in the way, nearer than the wall.
    assert metres[0] == pytest.approx(0.9, abs=0.02)


def test_synthetic_scan_is_never_chosen_automatically():
    """A robot must never act on invented ranges because a device was missing."""
    args = type("Args", (), {
        "lidar": "auto", "lidar_topic": "/nope", "lidar_port": "/dev/does-not-exist",
        "lidar_baud": robocam_client.LDS02_BAUD, "lidar_points": 360, "lidar_spin": "cw",
        "lidar_crc": "auto", "lidar_hz": 5.0,
    })()
    assert robocam_client.build_scan_source(args) is None
