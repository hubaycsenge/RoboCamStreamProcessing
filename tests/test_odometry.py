"""Odometry decoding.

The pose is three numbers that look fine whatever they contain, so almost
everything here is about the failures that do *not* announce themselves: a frame
name that means dead reckoning, a quaternion from an uninitialised filter, a
translation in millimetres.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robocam import odometry
from robocam.odometry import OdomError


def header(**overrides):
    base = {"type": "odom", "seq": 1, "x": 1.0, "y": 2.0, "yaw": 0.5, "frame": "map"}
    base.update(overrides)
    return base


def test_a_plain_pose_decodes():
    pose = odometry.decode_odom(header(), recv_ts_ns=99)
    assert (pose.x, pose.y) == (1.0, 2.0)
    assert pose.yaw == pytest.approx(0.5)
    assert pose.frame == "map"
    assert pose.recv_ts_ns == 99


def test_the_quaternion_wins_over_the_yaw_field():
    """One Euler convention in this system, and it is the quaternion's.

    A sender that computes its own yaw with a different convention would
    otherwise put a reconstruction in at the wrong heading, and nothing
    downstream could tell that from a robot that was really facing that way.
    """
    quat = _quat_from_yaw(1.25)
    pose = odometry.decode_odom(header(yaw=-3.0, qw=quat[0], qx=quat[1],
                                       qy=quat[2], qz=quat[3]))
    assert pose.yaw == pytest.approx(1.25)


def test_a_non_unit_quaternion_is_normalised_not_rejected():
    """A filter's output drifts from unit norm by parts per million.

    That is not an error and refusing it would reject a working robot; a
    quaternion of zeros is a different thing entirely and is refused below.
    """
    quat = [v * 1.0001 for v in _quat_from_yaw(0.4)]
    pose = odometry.decode_odom(header(qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3]))
    assert pose.yaw == pytest.approx(0.4, abs=1e-4)


def test_a_zero_quaternion_is_refused():
    with pytest.raises(OdomError, match="not a rotation"):
        odometry.decode_odom(header(qw=0.0, qx=0.0, qy=0.0, qz=0.0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_position_is_refused(bad):
    with pytest.raises(OdomError):
        odometry.decode_odom(header(x=bad))


def test_a_pose_in_millimetres_is_refused_as_out_of_range():
    """5 km from the origin is what metres read as millimetres looks like.

    The sanity limit exists because the alternative failure is silent: a cloud
    placed 5 km from the grid overlaps nothing, the comparison finds nothing,
    and the server reports healthily that there is nothing to report.
    """
    with pytest.raises(OdomError, match="sanity limit"):
        odometry.decode_odom(header(x=5_000_000.0))


def test_a_broken_velocity_does_not_cost_the_pose():
    """The pose is what the comparison needs; the twist is only reported."""
    pose = odometry.decode_odom(header(vx=float("nan")))
    assert pose.vx == 0.0
    assert pose.x == 1.0


def test_yaw_is_wrapped():
    pose = odometry.decode_odom(header(yaw=3.0 * math.pi))
    assert -math.pi < pose.yaw <= math.pi


def test_matrix2d_moves_a_point_into_the_pose_frame():
    pose = odometry.decode_odom(header(x=1.0, y=0.0, yaw=math.pi / 2))
    point = pose.matrix2d() @ np.array([2.0, 0.0, 1.0])
    # Two metres ahead of a robot facing +y, standing at (1, 0).
    assert point[0] == pytest.approx(1.0)
    assert point[1] == pytest.approx(2.0)


def test_analyse_measures_movement_from_the_previous_pose_not_the_twist():
    """What the reconstruction cares about is displacement, not reported speed.

    A source whose twist says 0.5 m/s while its poses do not move is a frozen
    odometry publisher, and that is exactly the case this distinguishes.
    """
    first = odometry.decode_odom(header(x=0.0, y=0.0, yaw=0.0), recv_ts_ns=0)
    second = odometry.decode_odom(header(x=0.3, y=0.4, yaw=0.0, vx=0.5),
                                  recv_ts_ns=1_000_000_000)
    summary = odometry.analyse(second, previous=first)
    assert summary["moved_m"] == pytest.approx(0.5)
    assert summary["measured_speed_ms"] == pytest.approx(0.5)


def test_analyse_wraps_the_turn():
    a = odometry.decode_odom(header(yaw=3.0), recv_ts_ns=0)
    b = odometry.decode_odom(header(yaw=-3.0), recv_ts_ns=1)
    turned = odometry.analyse(b, previous=a)["turned_deg"]
    # The short way round is ~16 degrees, not ~344.
    assert abs(turned) < 20.0


def test_wrap_angle_folds_into_the_atan2_interval():
    """Either sign of pi is acceptable; a *magnitude* of 3 pi is not."""
    assert abs(odometry.wrap_angle(3.0 * math.pi)) == pytest.approx(math.pi)
    assert abs(odometry.wrap_angle(-3.0 * math.pi)) == pytest.approx(math.pi)
    assert odometry.wrap_angle(0.5 * math.pi) == pytest.approx(0.5 * math.pi)


def test_compose_applies_a_correction_in_the_map_frame():
    pose = odometry.decode_odom(header(x=1.0, y=1.0, yaw=0.0))
    x, y, yaw = odometry.compose(pose, 0.1, -0.2, 0.05)
    assert (x, y) == pytest.approx((1.1, 0.8))
    assert yaw == pytest.approx(0.05)


def _quat_from_yaw(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
