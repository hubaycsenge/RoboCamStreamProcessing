"""The Compare box: the reconstruction against the robot's own grid.

The geometry is the part that can be quietly wrong, so most of this file builds a
synthetic scene with a known answer and checks the numbers come back.  The scene
is the one the whole stage exists for: a room the LiDAR mapped, with a table top
in it that a horizontal scanner at wheel height cannot see.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robocam.compare import (OPTICAL_TO_BODY, CameraMount, cloud_to_map, compare,
                             estimate_shift, occupancy_from_cloud)
from robocam.occupancy import Grid
from robocam.odometry import Odom


def a_grid(width=80, height=80, resolution=0.05, walls=True):
    """A 4 m square room, walls occupied, interior swept free."""
    cells = np.zeros((height, width), dtype=np.int8)
    if walls:
        cells[0, :] = 100
        cells[-1, :] = 100
        cells[:, 0] = 100
        cells[:, -1] = 100
    return Grid(cells=cells, resolution=resolution, origin=(0.0, 0.0, 0.0),
                frame="map", map_id="m1")


def identity_pose():
    """A camera pose that makes the model's world frame the camera frame."""
    return np.eye(4)


def test_a_camera_at_the_origin_looking_forward_puts_a_point_ahead_of_the_robot():
    """The convention flip, checked end to end rather than by inspection.

    The model works in the vision convention (x right, y down, z forward out of
    the lens); the robot works in REP-103 (x forward, y left, z up).  A point 2 m
    down the optical axis has to come out 2 m in front of the robot.
    """
    mount = CameraMount(x=0.0, y=0.0, z=0.0)
    odom = Odom(x=0.0, y=0.0, yaw=0.0, frame="map")
    point = np.array([[0.0, 0.0, 2.0]])          # 2 m along optical +z
    out = cloud_to_map(point, identity_pose(), odom, mount)
    assert out[0] == pytest.approx([2.0, 0.0, 0.0], abs=1e-9)


def test_optical_left_becomes_robot_left():
    mount = CameraMount(x=0.0, y=0.0, z=0.0)
    odom = Odom(x=0.0, y=0.0, yaw=0.0, frame="map")
    # -x in the optical frame is to the left of the image.
    out = cloud_to_map(np.array([[-1.0, 0.0, 0.0]]), identity_pose(), odom, mount)
    assert out[0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_optical_down_becomes_robot_down():
    mount = CameraMount(x=0.0, y=0.0, z=0.0)
    odom = Odom(x=0.0, y=0.0, yaw=0.0, frame="map")
    out = cloud_to_map(np.array([[0.0, 1.0, 0.0]]), identity_pose(), odom, mount)
    assert out[0] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_the_convention_matrix_is_a_rotation():
    assert np.linalg.det(OPTICAL_TO_BODY) == pytest.approx(1.0)
    assert (OPTICAL_TO_BODY @ OPTICAL_TO_BODY.T == pytest.approx(np.eye(3)))


def test_the_camera_mount_height_moves_the_cloud_up():
    odom = Odom(x=0.0, y=0.0, yaw=0.0, frame="map")
    point = np.array([[0.0, 0.0, 2.0]])
    low = cloud_to_map(point, identity_pose(), odom, CameraMount(x=0, y=0, z=0.0))
    high = cloud_to_map(point, identity_pose(), odom, CameraMount(x=0, y=0, z=0.45))
    assert high[0][2] - low[0][2] == pytest.approx(0.45)


def test_the_robots_yaw_rotates_the_cloud():
    mount = CameraMount(x=0.0, y=0.0, z=0.0)
    odom = Odom(x=0.0, y=0.0, yaw=math.pi / 2, frame="map")
    out = cloud_to_map(np.array([[0.0, 0.0, 2.0]]), identity_pose(), odom, mount)
    # Facing +y, so 2 m ahead is (0, 2).
    assert out[0][:2] == pytest.approx([0.0, 2.0], abs=1e-9)


def test_the_model_pose_is_inverted_not_applied():
    """A camera that has moved forward in the model's world sees points behind it.

    Getting this backwards doubles every translation instead of cancelling it,
    and the symptom is a reconstruction that drifts away at twice the robot's
    speed — plausible-looking motion, entirely wrong scale of error.
    """
    mount = CameraMount(x=0.0, y=0.0, z=0.0)
    odom = Odom(x=0.0, y=0.0, yaw=0.0, frame="map")
    pose = np.eye(4)
    pose[:3, 3] = (0.0, 0.0, 1.0)     # camera 1 m along the model's +z
    # A point at the model's origin is then 1 m *behind* the camera.
    out = cloud_to_map(np.array([[0.0, 0.0, 0.0]]), pose, odom, mount)
    assert out[0] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)


def test_the_height_slice_drops_the_floor_and_the_ceiling():
    grid = a_grid()
    points = np.array([
        [1.0, 1.0, 0.0],     # floor
        [1.0, 1.0, 0.8],     # table top
        [1.0, 1.0, 2.5],     # ceiling
    ])
    occupied, stats = occupancy_from_cloud(points, grid, z_min=0.05, z_max=1.6,
                                           min_points=1)
    assert stats["in_slice"] == 1
    assert stats["below_slice"] == 1 and stats["above_slice"] == 1
    assert occupied.sum() == 1


def test_a_single_point_is_not_a_surface():
    """One point in a cell is one pixel of a monocular depth estimate."""
    grid = a_grid()
    points = np.array([[1.0, 1.0, 0.8]])
    occupied, _ = occupancy_from_cloud(points, grid, min_points=3)
    assert occupied.sum() == 0


def test_points_outside_the_grid_are_dropped_not_piled_on_the_border():
    grid = a_grid()
    points = np.tile(np.array([[100.0, 100.0, 0.8]]), (10, 1))
    occupied, stats = occupancy_from_cloud(points, grid, min_points=1)
    assert stats["in_bounds"] == 0
    assert occupied.sum() == 0


def test_compare_refuses_to_cross_frames():
    """The one refusal.  A pose in odom against a grid in map is not comparable.

    They differ by SLAM's accumulated correction, so the comparison would
    produce a confident wrong answer rather than an obviously broken one.
    """
    grid = a_grid()
    odom = Odom(x=1.0, y=1.0, yaw=0.0, frame="odom")
    result = compare(np.zeros((10, 3)), identity_pose(), odom, grid, CameraMount())
    assert not result.ok
    assert "refusing to compare across frames" in result.stats["error"]
    assert result.patch is None


def test_compare_finds_a_table_the_scanner_could_not_see():
    """The whole point of the stage, as an end-to-end assertion.

    The grid says the middle of the room is swept free — which is true at the
    scanner's own height. The cloud has a surface at 75 cm there. What comes back
    is a patch marking those cells occupied, and ``over_free`` counting them:
    cells the scanner *looked at and found empty*, which is exactly the
    out-of-plane obstacle this exists to find.
    """
    grid = a_grid()
    odom = Odom(x=0.5, y=2.0, yaw=0.0, frame="map")
    mount = CameraMount(x=0.0, y=0.0, z=0.5)
    # A 40 cm square of table top, 1.5 m in front of the robot, at 75 cm.
    xs, ys = np.meshgrid(np.linspace(1.8, 2.2, 20), np.linspace(1.8, 2.2, 20))
    table = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.75)])
    points = _to_optical(table, odom, mount)

    result = compare(points, identity_pose(), odom, grid, mount, min_points=1)

    assert result.ok
    assert result.stats["new_cells"] > 20
    assert result.stats["over_free"] == result.stats["new_cells"]
    assert result.patch is not None
    assert (result.patch.cells == 100).sum() == result.stats["new_cells"]


def test_compare_reports_agreement_when_the_cloud_matches_the_walls():
    """Agreement is the health number: a broken placement drives it to zero."""
    grid = a_grid()
    odom = Odom(x=2.0, y=2.0, yaw=0.0, frame="map")
    mount = CameraMount(x=0.0, y=0.0, z=0.5)
    # Points along the mapped wall at x = 0 (column 0 of the grid).
    wall = np.column_stack([
        np.full(60, 0.02), np.linspace(0.5, 3.5, 60), np.full(60, 0.6),
    ])
    result = compare(_to_optical(wall, odom, mount), identity_pose(), odom, grid,
                     mount, min_points=1)
    assert result.stats["agreement"] > 0.9


def test_a_misplaced_cloud_shows_up_as_near_zero_agreement():
    grid = a_grid()
    odom = Odom(x=2.0, y=2.0, yaw=0.0, frame="map")
    mount = CameraMount(x=0.0, y=0.0, z=0.5)
    interior = np.column_stack([
        np.linspace(1.0, 3.0, 60), np.full(60, 2.0), np.full(60, 0.6),
    ])
    result = compare(_to_optical(interior, odom, mount), identity_pose(), odom,
                     grid, mount, min_points=1)
    assert result.stats["agreement"] < 0.1


def test_compare_never_clears_a_cell_the_scanner_marked_occupied():
    """``missing`` is reported and never acted on."""
    grid = a_grid()
    odom = Odom(x=2.0, y=2.0, yaw=0.0, frame="map")
    mount = CameraMount(x=0.0, y=0.0, z=0.5)
    table = np.column_stack([
        np.full(40, 2.0), np.linspace(1.9, 2.1, 40), np.full(40, 0.75),
    ])
    result = compare(_to_optical(table, odom, mount), identity_pose(), odom, grid,
                     mount, min_points=1)
    assert result.stats["missing"] > 0
    if result.updated is not None:
        # Every wall cell is still occupied in the updated grid.
        assert (result.updated[0, :] == 100).all()


def test_a_handful_of_scattered_cells_does_not_earn_a_patch():
    grid = a_grid()
    odom = Odom(x=2.0, y=2.0, yaw=0.0, frame="map")
    mount = CameraMount(x=0.0, y=0.0, z=0.5)
    speck = np.array([[2.5, 2.0, 0.75]])
    result = compare(_to_optical(speck, odom, mount), identity_pose(), odom, grid,
                     mount, min_points=1, min_new_cells=4)
    assert result.patch is None
    assert result.stats["patched"] is False


def test_estimate_shift_finds_a_known_offset():
    grid = a_grid()
    occupied_map, _, _ = grid.classify()
    # The same walls, slid two cells right and one down.
    cloud = np.roll(np.roll(occupied_map, -1, axis=0), -2, axis=1)
    shift = estimate_shift(cloud, grid, search_cells=4, min_inliers=10)
    assert shift is not None
    assert shift["dx"] == pytest.approx(2 * grid.resolution)
    assert shift["dy"] == pytest.approx(1 * grid.resolution)


def test_estimate_shift_declines_when_the_cloud_already_agrees():
    """No offset is better than zero, so there is nothing to offer."""
    grid = a_grid()
    occupied_map, _, _ = grid.classify()
    assert estimate_shift(occupied_map, grid, min_inliers=10) is None


def test_estimate_shift_declines_on_too_little_overlap():
    """A hint from a handful of inliers is a histogram artefact."""
    grid = a_grid()
    cloud = np.zeros(grid.cells.shape, dtype=bool)
    cloud[5, 5] = True
    assert estimate_shift(cloud, grid, min_inliers=40) is None


def _to_optical(points_map: np.ndarray, odom: Odom, mount: CameraMount) -> np.ndarray:
    """Inverse of cloud_to_map with an identity model pose.

    The tests describe scenes in the map frame because that is where their
    intent is legible; the processor hands ``compare`` points in the model's
    frame.  This is the bridge, and it being an exact inverse is what makes the
    assertions above about the map frame mean anything.
    """
    t_map_cam = _t_map_cam(odom, mount)
    rot, trans = t_map_cam[:3, :3], t_map_cam[:3, 3]
    return (np.asarray(points_map, dtype=np.float64) - trans) @ rot


def _t_map_cam(odom: Odom, mount: CameraMount) -> np.ndarray:
    c, s = math.cos(odom.yaw), math.sin(odom.yaw)
    t_map_base = np.eye(4)
    t_map_base[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t_map_base[:3, 3] = (odom.x, odom.y, odom.z)
    return t_map_base @ mount.matrix()
