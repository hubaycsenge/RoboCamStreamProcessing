"""LiDAR decoding, analysis and projection.

The geometry tests are written against scans built by hand with returns in
known directions, because the failure this file exists to catch is a sign or a
convention error — a scan that is mirrored, or rotated 90°, still produces
entirely plausible numbers and would go unnoticed until the robot turned the
wrong way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robocam import lidar, wire


def build_scan(ranges_m, **header_overrides):
    """Encode ranges as a client would, then decode them as the server does."""
    header = wire.scan(seq=1, count=len(ranges_m), t_capture_ns=0,
                       range_min=0.12, range_max=12.0)
    header.update(header_overrides)
    payload = lidar.encode_scan_payload(ranges_m)
    return lidar.decode_scan(header, payload, recv_ts_ns=123)


def uniform(count=360, value=3.0):
    return np.full(count, value, dtype=np.float32)


def at_bearing(deg, value_m, count=360, background=np.nan):
    """A scan of no-returns except one at the given bearing (CCW, 0 = forward)."""
    ranges = np.full(count, background, dtype=np.float32)
    ranges[int(round(deg % 360.0 / 360.0 * count)) % count] = value_m
    return ranges


# -- decoding ---------------------------------------------------------------


def test_roundtrip_preserves_millimetre_ranges():
    values = np.array([0.5, 1.234, 12.0, 0.123], dtype=np.float32)
    scan = build_scan(values)
    assert scan.count == 4
    np.testing.assert_allclose(scan.ranges, values, atol=0.001)


def test_zero_means_no_return_not_zero_metres():
    """A no-return must not read as an obstacle against the sensor's face."""
    scan = build_scan([0.0, 2.0, 0.0, 2.0])
    assert np.isnan(scan.ranges[0])
    assert int(scan.valid.sum()) == 2
    assert scan.summary == {}  # filled by the server, not the decoder


def test_returns_outside_the_device_limits_are_discarded():
    scan = build_scan([0.05, 3.0, 20.0], range_min=0.12, range_max=12.0)
    assert np.isnan(scan.ranges[0]), "too close to be real"
    assert np.isnan(scan.ranges[2]), "beyond the device's range"
    assert scan.ranges[1] == pytest.approx(3.0, abs=0.001)


def test_float32_encoding_treats_inf_and_nan_as_gaps():
    """ROS publishes inf for 'nothing there' and nan for 'no reading'."""
    values = np.array([1.5, np.inf, np.nan, 2.5], dtype=np.float32)
    header = wire.scan(seq=0, count=4, t_capture_ns=0, encoding=wire.SCAN_ENC_F32_M)
    scan = lidar.decode_scan(header, values.tobytes())
    assert list(np.isfinite(scan.ranges)) == [True, False, False, True]


def test_intensities_are_decoded_when_declared():
    ranges = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    payload = lidar.encode_scan_payload(ranges, intensities=[10, 20, 30])
    header = wire.scan(seq=0, count=3, t_capture_ns=0, intensities=True)
    scan = lidar.decode_scan(header, payload)
    assert list(scan.intensities) == [10, 20, 30]


def test_default_angle_increment_is_a_full_revolution():
    scan = build_scan(uniform(360))
    assert scan.angle_increment == pytest.approx(2 * math.pi / 360)
    # Bearings wrap into (-pi, pi] rather than running 0..2pi.
    assert scan.bearings.min() >= -math.pi
    assert scan.bearings.max() <= math.pi


def test_negative_increment_reverses_the_spin_direction():
    """A clockwise device declares it with the sign, not by reversing the array."""
    cw = build_scan(uniform(4), angle_increment=-2 * math.pi / 4)
    # Point 1 of a clockwise device is 90° clockwise, i.e. bearing -90°.
    assert math.degrees(cw.bearings[1]) == pytest.approx(-90.0, abs=0.01)


@pytest.mark.parametrize("header, payload, message", [
    ({"count": 0}, b"", "no points"),
    ({"count": 100000}, b"", "limit"),
    ({"count": 10}, b"\x00\x01", "expected at least"),
    ({"count": 4, "encoding": "polar_ascii"}, b"x" * 8, "unsupported scan encoding"),
])
def test_malformed_scans_raise_rather_than_return_nonsense(header, payload, message):
    full = wire.scan(seq=0, count=1, t_capture_ns=0)
    full.update(header)
    with pytest.raises(lidar.ScanError) as exc:
        lidar.decode_scan(full, payload)
    assert message in str(exc.value)


# -- analysis ---------------------------------------------------------------


def test_nearest_reports_range_and_bearing():
    scan = build_scan(at_bearing(90, 0.8, background=5.0))
    summary = lidar.analyse(scan)
    assert summary["nearest_m"] == pytest.approx(0.8, abs=0.01)
    assert summary["nearest_deg"] == pytest.approx(90.0, abs=1.0)


def test_obstacle_flag_only_fires_for_the_front_arc():
    """Something 30 cm behind the robot is not something to stop for."""
    behind = lidar.analyse(build_scan(at_bearing(180, 0.3, background=5.0)), obstacle_m=0.5)
    ahead = lidar.analyse(build_scan(at_bearing(0, 0.3, background=5.0)), obstacle_m=0.5)
    assert behind["obstacle"] is False
    assert ahead["obstacle"] is True
    assert ahead["front_min_m"] == pytest.approx(0.3, abs=0.01)


def test_mount_yaw_rotates_the_reported_bearings():
    """A scanner mounted backwards must not report obstacles ahead."""
    scan = build_scan(at_bearing(180, 0.3, background=5.0))
    rotated = lidar.analyse(scan, mount_yaw_deg=180.0, obstacle_m=0.5)
    assert rotated["nearest_deg"] == pytest.approx(0.0, abs=1.0)
    assert rotated["obstacle"] is True


def test_sector_minima_land_in_the_right_sector():
    scan = build_scan(at_bearing(0, 1.0, background=np.nan))
    summary = lidar.analyse(scan, sectors=12)
    minima = summary["sector_min_m"]
    assert len(minima) == 12
    # Sector 6 of 12 is the one centred on straight ahead.
    assert minima[6] == pytest.approx(1.0, abs=0.01)
    assert all(v is None for i, v in enumerate(minima) if i != 6)


def test_free_direction_points_at_the_gap():
    """A wall everywhere except a doorway: the robot should be sent at it.

    The gap is 31° — narrower than one 30° sector once it straddles a sector
    boundary — so this also pins down that the search runs at the scan's own
    resolution.  Quantising to sectors would report the only way out as blocked.
    """
    ranges = np.full(360, 0.4, dtype=np.float32)
    ranges[75:106] = 6.0                      # a 31 deg gap centred on 90 deg
    summary = lidar.analyse(build_scan(ranges), sectors=12, clear_m=1.0)
    assert summary["free_deg"] == pytest.approx(90.0, abs=2.0)
    assert summary["free_width_deg"] == pytest.approx(31.0, abs=1.5)


def test_free_direction_can_wrap_around_the_back():
    ranges = np.full(360, 0.4, dtype=np.float32)
    ranges[170:191] = 6.0                     # gap straddling the +/-180 seam
    summary = lidar.analyse(build_scan(ranges), clear_m=1.0, min_free_deg=15.0)
    assert abs(summary["free_deg"]) == pytest.approx(180.0, abs=2.0)


def test_a_gap_too_narrow_to_drive_through_is_not_a_free_direction():
    """One long reading between two walls is a measurement, not a doorway."""
    ranges = np.full(360, 0.4, dtype=np.float32)
    ranges[90] = 6.0
    summary = lidar.analyse(build_scan(ranges), clear_m=1.0, min_free_deg=15.0)
    assert summary["free_deg"] is None
    assert summary["free_width_deg"] < 15.0


def test_a_scan_of_no_returns_is_reported_as_blind_not_as_clear():
    """The dangerous failure: a dead scanner must not read as an empty room."""
    summary = lidar.analyse(build_scan(np.zeros(360, dtype=np.float32)))
    assert summary["blind"] is True
    assert summary["valid"] == 0
    assert summary["coverage"] == 0.0
    assert summary["nearest_m"] is None
    assert summary["obstacle"] is False


def test_coverage_notices_a_mostly_empty_scan():
    ranges = np.zeros(360, dtype=np.float32)
    ranges[:36] = 2.0
    summary = lidar.analyse(build_scan(ranges))
    assert summary["coverage"] == pytest.approx(0.1, abs=0.01)


def test_summary_is_json_serialisable():
    """It travels in a result header, so NaN and numpy scalars are not allowed."""
    import json

    summary = lidar.analyse(build_scan(at_bearing(45, 1.0)))
    text = json.dumps(summary, allow_nan=False)
    assert "NaN" not in text


# -- projection into the camera --------------------------------------------


def test_forward_bearing_maps_to_the_image_centre():
    cols = lidar.bearing_to_column(np.array([0.0]), width=640, hfov_deg=70.0)
    assert cols[0] == pytest.approx(320.0, abs=0.5)


def test_column_mapping_is_tangent_not_linear():
    """A linear mapping is wrong by several percent at the edges of the frame."""
    half = math.radians(70.0) / 2.0
    # Half way to the edge of the field, on the right of the image.
    col = float(lidar.bearing_to_column(np.array([-half / 2]), width=640, hfov_deg=70.0)[0])
    linear = 320.0 + 320.0 * 0.5
    assert col != pytest.approx(linear, abs=1.0)
    assert 320.0 < col < linear


def test_bearings_behind_the_camera_are_not_projected():
    cols = lidar.bearing_to_column(np.array([math.pi, math.pi / 2]), width=640, hfov_deg=70.0)
    assert np.isnan(cols).all()


def test_left_of_the_robot_is_left_of_the_image():
    """The mirror check. A positive bearing is CCW, which is the left of frame."""
    scan = build_scan(at_bearing(20, 2.0))
    bins = lidar.project_columns(scan, width=640, hfov_deg=70.0, bins=8)
    hit = int(np.nanargmin(np.where(np.isnan(bins), np.inf, bins)))
    assert hit < 4, "a return 20 deg counter-clockwise belongs in the left half"


def test_project_columns_leaves_unseen_directions_as_gaps():
    scan = build_scan(at_bearing(0, 2.0))
    bins = lidar.project_columns(scan, width=640, hfov_deg=70.0, bins=16)
    assert np.isfinite(bins).sum() == 1
    assert np.isnan(bins).sum() == 15


def test_range_for_box_returns_metres_for_the_columns_it_covers():
    ranges = np.full(360, np.nan, dtype=np.float32)
    ranges[0] = 1.5           # straight ahead -> image centre
    ranges[40] = 4.0          # 40 deg away -> outside a 70 deg field
    scan = build_scan(ranges)

    assert lidar.range_for_box(scan, 300, 340, width=640) == pytest.approx(1.5, abs=0.01)
    # A box on the far left covers no beam at all.
    assert lidar.range_for_box(scan, 0, 40, width=640) is None


def test_range_for_box_min_and_median_answer_different_questions():
    ranges = np.full(360, np.nan, dtype=np.float32)
    for i, value in ((357, 5.0), (0, 1.0), (3, 5.0)):
        ranges[i] = value
    scan = build_scan(ranges)

    assert lidar.range_for_box(scan, 0, 640, width=640, reducer="min") == pytest.approx(1.0)
    assert lidar.range_for_box(scan, 0, 640, width=640, reducer="median") == pytest.approx(5.0)


def test_to_xy_puts_forward_on_x_and_left_on_y():
    scan = build_scan(at_bearing(90, 2.0))
    xs, ys = lidar.to_xy(scan)
    assert xs[0] == pytest.approx(0.0, abs=0.01)
    assert ys[0] == pytest.approx(2.0, abs=0.01)
