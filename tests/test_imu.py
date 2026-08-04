"""IMU decoding, unit handling and attitude analysis.

The attitude tests are written against quaternions and accelerometer vectors
built by hand for known orientations, because the failure this file exists to
catch is the same one the LiDAR's geometry tests exist to catch: a sign or an
axis convention error produces entirely plausible numbers.  A board mounted on
its side reports a confident, level-looking attitude that is wrong by 90°, and
nothing downstream can tell.

The other half is units.  A gyro read as rad/s when it is deg/s is wrong by 57x
and still looks like a robot turning, so the decoder is tested to convert what
the client declared and to refuse what it does not recognise.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from robocam import imu, wire


def build_batch(values, offsets_us=None, **header_overrides):
    """Encode samples as a client would, then decode them as the server does."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    fields = header_overrides.pop("fields", wire.IMU_FIELDS)
    header = wire.imu(seq=1, count=arr.shape[0], t_capture_ns=0, fields=fields)
    header.update(header_overrides)
    payload = imu.encode_imu_payload(arr, offsets_us)
    return imu.decode_imu(header, payload, recv_ts_ns=123)


def level_sample(**over):
    """One sample of a level robot at rest: gravity on +z, no rotation."""
    sample = {"wx": 0.0, "wy": 0.0, "wz": 0.0,
              "ax": 0.0, "ay": 0.0, "az": imu.G_MS2,
              "mx": 20.0, "my": -5.0, "mz": 40.0,
              "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
    sample.update(over)
    return [sample[name] for name in wire.IMU_FIELDS]


def quat_for(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    """``(w, x, y, z)`` for intrinsic Z-Y-X, the convention the server decodes."""
    r, p, y = (math.radians(v) / 2 for v in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


# -- decoding ---------------------------------------------------------------


def test_roundtrip_preserves_the_channels():
    batch = build_batch([level_sample(), level_sample(wx=0.5)])
    assert batch.count == 2
    assert batch.fields == wire.IMU_FIELDS
    np.testing.assert_allclose(batch.column("wx"), [0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(batch.column("az"), [imu.G_MS2] * 2, atol=1e-4)
    assert batch.summary == {}  # filled by the server, not the decoder


def test_degrees_per_second_are_converted_to_radians():
    """The 57x mistake, caught at the only place it can be caught."""
    batch = build_batch([level_sample(wz=180.0)], gyro_units="deg/s")
    np.testing.assert_allclose(batch.column("wz"), [math.pi], rtol=1e-5)


def test_g_is_converted_to_metres_per_second_squared():
    batch = build_batch([level_sample(az=1.0)], accel_units="g")
    np.testing.assert_allclose(batch.column("az"), [imu.G_MS2], rtol=1e-5)


def test_units_already_si_are_left_alone():
    batch = build_batch([level_sample(wz=1.0)], gyro_units="rad/s", accel_units="m/s2")
    np.testing.assert_allclose(batch.column("wz"), [1.0], rtol=1e-6)
    np.testing.assert_allclose(batch.column("az"), [imu.G_MS2], rtol=1e-5)


@pytest.mark.parametrize("units, message", [
    ({"gyro_units": "rpm"}, "gyro_units"),
    ({"accel_units": "mg"}, "accel_units"),
])
def test_an_unrecognised_unit_is_refused_rather_than_assumed(units, message):
    with pytest.raises(imu.ImuError, match=message):
        build_batch([level_sample()], **units)


def test_a_client_without_a_magnetometer_sends_a_shorter_row():
    """Absent is not zero: a missing channel must not read as a measured zero."""
    fields = ("wx", "wy", "wz", "ax", "ay", "az", "qw", "qx", "qy", "qz")
    batch = build_batch([[0.0, 0.0, 0.0, 0.0, 0.0, imu.G_MS2, 1.0, 0.0, 0.0, 0.0]],
                        fields=fields)
    assert batch.mag is None
    assert batch.gyro is not None and batch.quat is not None


def test_offsets_give_the_measured_rate():
    """The rate the sensor delivered, not the rate it claims."""
    offsets = [0, 10_000, 20_000, 30_000]  # 10 ms apart -> 100 Hz
    batch = build_batch([level_sample() for _ in range(4)], offsets_us=offsets,
                        rate_hz=100.0)
    assert batch.rate_hz == pytest.approx(100.0)
    assert batch.span_s == pytest.approx(0.03)
    assert batch.declared_rate_hz == pytest.approx(100.0)


def test_a_single_sample_burst_reports_no_rate_rather_than_dividing_by_zero():
    batch = build_batch([level_sample()])
    assert batch.span_s == 0.0
    assert batch.rate_hz == 0.0


@pytest.mark.parametrize("overrides, payload, message", [
    ({"count": 0}, b"", "no samples"),
    ({"count": imu.MAX_SAMPLES + 1}, b"", "exceeds"),
    ({"count": 4}, b"\x00" * 8, "expected at least"),
    ({"encoding": "f64"}, b"", "encoding"),
    ({"fields": ["wx", "wx"]}, b"", "duplicates"),
    ({"fields": "wx,wy"}, b"", "non-empty list"),
])
def test_malformed_bursts_raise_rather_than_return_nonsense(overrides, payload, message):
    header = wire.imu(seq=1, count=1, t_capture_ns=0)
    header.update(overrides)
    with pytest.raises(imu.ImuError, match=message):
        imu.decode_imu(header, payload)


@pytest.mark.parametrize("fields", [None, []])
def test_an_unstated_field_list_means_the_protocol_default(fields):
    """Absent and empty both mean "the standard 13 channels", not "no channels"."""
    header = wire.imu(seq=1, count=1, t_capture_ns=0)
    header["fields"] = fields
    batch = imu.decode_imu(header, imu.encode_imu_payload([level_sample()]))
    assert batch.fields == wire.IMU_FIELDS


# -- attitude ---------------------------------------------------------------


@pytest.mark.parametrize("roll, pitch, yaw", [
    (0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, -20.0, 0.0),
    (0.0, 0.0, 90.0), (10.0, 15.0, -45.0),
])
def test_euler_roundtrips_through_the_quaternion(roll, pitch, yaw):
    got = imu.quat_to_euler_deg(quat_for(roll, pitch, yaw))
    assert got == pytest.approx((roll, pitch, yaw), abs=1e-4)


def test_pitch_at_the_vertical_singularity_clamps_instead_of_returning_nan():
    """A quaternion a rounding error past vertical must not produce NaN."""
    _, pitch, _ = imu.quat_to_euler_deg(quat_for(0.0, 90.0, 0.0))
    assert pitch == pytest.approx(90.0, abs=1e-3)


@pytest.mark.parametrize("roll, pitch, expected", [
    (0.0, 0.0, 0.0), (30.0, 0.0, 30.0), (0.0, -25.0, 25.0), (90.0, 0.0, 90.0),
])
def test_tilt_is_the_angle_off_level_whichever_way_it_leans(roll, pitch, expected):
    assert imu.tilt_from_quat_deg(quat_for(roll, pitch)) == pytest.approx(expected, abs=1e-3)


def test_tilt_ignores_yaw():
    """Spinning on the spot is not tipping over."""
    assert imu.tilt_from_quat_deg(quat_for(0.0, 0.0, 145.0)) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("q", [
    (float("nan"),) * 4, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
])
def test_a_placeholder_quaternion_is_not_treated_as_level(q):
    """All-zeros and NaN are what an unconverged filter emits, not an attitude."""
    assert not imu._quat_is_usable(q)


def test_a_slightly_unnormalised_quaternion_is_still_usable():
    assert imu._quat_is_usable((1.02, 0.0, 0.0, 0.0))


# -- analysis ---------------------------------------------------------------


def test_a_level_robot_at_rest_is_still():
    summary = imu.analyse(build_batch([level_sample() for _ in range(10)]))
    assert summary["still"] is True
    assert summary["turning"] is False
    assert summary["shaking"] is False
    assert summary["tilted"] is False
    assert summary["gravity_ok"] is True
    assert summary["tilt_deg"] == pytest.approx(0.0, abs=0.01)


def test_a_steady_turn_reports_the_yaw_rate_and_is_not_still():
    rate = math.radians(45.0)
    summary = imu.analyse(build_batch([level_sample(wz=rate) for _ in range(10)]))
    assert summary["yaw_rate_dps"] == pytest.approx(45.0, abs=0.1)
    assert summary["turning"] is True
    assert summary["still"] is False


def test_constant_velocity_still_reads_as_still():
    """Documented and deliberate: an accelerometer cannot see constant speed."""
    summary = imu.analyse(build_batch([level_sample() for _ in range(10)]))
    assert summary["still"] is True


def test_the_max_rate_survives_the_averaging():
    """A jolt inside a burst is exactly what the mean is designed to hide."""
    samples = [level_sample() for _ in range(9)] + [level_sample(wx=math.radians(200.0))]
    summary = imu.analyse(build_batch(samples))
    assert summary["gyro_max_dps"] == pytest.approx(200.0, abs=0.5)
    assert abs(summary["roll_rate_dps"]) < 25.0


def test_an_impact_is_flagged_as_shock():
    samples = [level_sample() for _ in range(9)] + [level_sample(az=40.0)]
    summary = imu.analyse(build_batch(samples))
    assert summary["shock"] is True
    assert summary["accel_mag_max"] == pytest.approx(40.0, abs=0.01)


def test_vibration_shows_up_as_shaking():
    samples = [level_sample(az=imu.G_MS2 + (2.0 if i % 2 else -2.0)) for i in range(10)]
    summary = imu.analyse(build_batch(samples))
    assert summary["shaking"] is True
    assert summary["still"] is False


def test_a_units_mistake_shows_up_as_gravity_not_ok():
    """Accelerometer left in g: the mean is 1.0, not 9.81."""
    summary = imu.analyse(build_batch([level_sample(az=1.0) for _ in range(5)]))
    assert summary["gravity_ok"] is False


def test_attitude_prefers_the_quaternion():
    q = quat_for(roll_deg=30.0)
    sample = level_sample(qw=q[0], qx=q[1], qy=q[2], qz=q[3])
    summary = imu.analyse(build_batch([sample]))
    assert summary["attitude_from"] == "quaternion"
    assert summary["roll_deg"] == pytest.approx(30.0, abs=0.01)
    assert summary["tilt_deg"] == pytest.approx(30.0, abs=0.01)


def test_attitude_falls_back_to_the_accelerometer_without_a_quaternion():
    """A 30° roll puts gravity on y as well as z; that is recoverable."""
    fields = ("wx", "wy", "wz", "ax", "ay", "az")
    roll = math.radians(30.0)
    row = [0.0, 0.0, 0.0, 0.0, imu.G_MS2 * math.sin(roll), imu.G_MS2 * math.cos(roll)]
    summary = imu.analyse(build_batch([row], fields=fields))
    assert summary["attitude_from"] == "accelerometer"
    assert summary["roll_deg"] == pytest.approx(30.0, abs=0.05)
    assert summary["tilt_deg"] == pytest.approx(30.0, abs=0.05)
    # No magnetometer and no gyro can supply an absolute heading.
    assert summary["yaw_deg"] is None


def test_a_nan_quaternion_falls_back_rather_than_reporting_level():
    """The worst failure mode: an unconverged filter reading as confidently level."""
    nan = float("nan")
    sample = level_sample(qw=nan, qx=nan, qy=nan, qz=nan)
    summary = imu.analyse(build_batch([sample]))
    assert summary["attitude_from"] == "accelerometer"
    assert summary["tilt_deg"] == pytest.approx(0.0, abs=0.05)


def test_the_accelerometer_fallback_is_refused_when_it_is_not_gravity():
    """Under acceleration, "down" is contaminated — better no attitude than a wrong one."""
    fields = ("ax", "ay", "az")
    summary = imu.analyse(build_batch([[0.0, 0.0, 1.0]], fields=fields))
    assert summary["attitude_from"] is None
    assert summary["tilt_deg"] is None


def test_a_tilt_past_the_threshold_is_flagged():
    q = quat_for(pitch_deg=25.0)
    sample = level_sample(qw=q[0], qx=q[1], qy=q[2], qz=q[3])
    summary = imu.analyse(build_batch([sample]), tilt_warn_deg=15.0)
    assert summary["tilted"] is True


def test_a_board_mounted_on_its_side_shows_up_immediately():
    """The mounting mistake the module docstring warns about, as a permanent tilt."""
    q = quat_for(roll_deg=90.0)
    sample = level_sample(qw=q[0], qx=q[1], qy=q[2], qz=q[3])
    summary = imu.analyse(build_batch([sample]))
    assert summary["tilt_deg"] == pytest.approx(90.0, abs=0.05)
    assert summary["tilted"] is True


def test_missing_channels_are_reported_as_absent_not_as_zero():
    summary = imu.analyse(build_batch([[0.0, 0.0, 0.0]], fields=("wx", "wy", "wz")))
    assert summary["gyro"] is True
    assert summary["accel"] is False
    assert summary["mag"] is False
    assert summary["accel_mag_mean"] is None
    # Neither half of "still" is known, so it must not claim the robot is still.
    assert summary["still"] is False


def test_non_finite_samples_are_masked_rather_than_poisoning_the_mean():
    good = level_sample(wz=math.radians(10.0))
    bad = level_sample(wz=float("nan"))
    summary = imu.analyse(build_batch([good, bad, good]))
    assert summary["yaw_rate_dps"] == pytest.approx(10.0, abs=0.05)


def test_the_dropped_count_travels_with_the_burst():
    """A gap the client knows about, stated rather than inferred from a low rate."""
    batch = build_batch([level_sample()], dropped=17)
    assert imu.analyse(batch)["dropped"] == 17


def test_summary_is_json_serialisable():
    """NaN is not valid JSON, and a bare NaN token is what strict parsers reject."""
    summary = imu.analyse(build_batch([level_sample(), level_sample(wz=float("nan"))]))
    text = json.dumps(summary)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["samples"] == 2


def test_yaw_is_never_claimed_to_be_absolute():
    """Documented as drift, and the flag must say so however attitude was found."""
    assert imu.analyse(build_batch([level_sample()]))["yaw_absolute"] is False
