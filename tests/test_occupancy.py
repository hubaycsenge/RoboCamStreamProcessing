"""The occupancy grid: decoding, geometry, and the merge rule.

The merge rule gets the most attention here because it is the one place where
being wrong is unsafe rather than merely wrong: the server may add obstacles to
the robot's map and must never clear one.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

from robocam import occupancy, wire
from robocam.occupancy import Grid, MapError


def grid_header(width=4, height=3, **overrides):
    base = {
        "type": "map", "seq": 1, "encoding": wire.MAP_ENC_I8,
        "width": width, "height": height, "resolution": 0.05,
        "origin": [-1.0, -2.0, 0.0], "frame": "map", "map_id": "m1",
        "occupied_min": 65, "free_max": 25, "full": True,
    }
    base.update(overrides)
    return base


def cells(width=4, height=3, fill=0):
    return np.full((height, width), fill, dtype=np.int8)


def test_a_raw_grid_round_trips():
    data = cells()
    data[1, 2] = 100
    grid = occupancy.decode_map(grid_header(), data.tobytes())
    assert grid.width == 4 and grid.height == 3
    assert grid.cells[1, 2] == 100
    assert grid.map_id == "m1"


def test_a_zlib_grid_round_trips():
    data = cells()
    data[0, 0] = 77
    payload = occupancy.encode_cells(data, wire.MAP_ENC_I8_ZLIB)
    grid = occupancy.decode_map(grid_header(encoding=wire.MAP_ENC_I8_ZLIB), payload)
    assert grid.cells[0, 0] == 77


def test_zlib_actually_helps_on_a_realistic_grid():
    """The reason the compressed encoding exists at all.

    A real room is mostly unknown and mostly runs of equal values, which is what
    makes uploading it every few seconds affordable.
    """
    room = np.full((400, 400), -1, dtype=np.int8)
    room[100:300, 100] = 100
    room[100:300, 300] = 100
    room[100, 100:300] = 100
    room[101:299, 101:299] = 0
    packed = occupancy.encode_cells(room, wire.MAP_ENC_I8_ZLIB)
    assert len(packed) < room.size // 20


def test_a_payload_of_the_wrong_length_is_refused():
    with pytest.raises(MapError, match="bytes for a"):
        occupancy.decode_map(grid_header(), cells().tobytes()[:-1])


def test_a_grid_over_the_cell_limit_is_refused_before_allocating():
    with pytest.raises(MapError, match="cell limit"):
        occupancy.decode_map(grid_header(width=10_000, height=10_000), b"",
                             max_cells=1000)


def test_a_compressed_bomb_cannot_expand_past_the_declared_grid():
    """The decompression is bounded by what the header claims, not by trust."""
    payload = zlib.compress(np.zeros(1_000_000, dtype=np.int8).tobytes())
    with pytest.raises(MapError):
        occupancy.decode_map(grid_header(encoding=wire.MAP_ENC_I8_ZLIB), payload)


@pytest.mark.parametrize("resolution", [0.0, 0.0001, 50.0])
def test_an_implausible_resolution_is_refused(resolution):
    with pytest.raises(MapError, match="resolution"):
        occupancy.decode_map(grid_header(resolution=resolution), cells().tobytes())


def test_thresholds_that_cross_are_refused():
    with pytest.raises(MapError, match="free_max"):
        occupancy.decode_map(grid_header(occupied_min=20, free_max=50),
                             cells().tobytes())


def test_world_and_cell_coordinates_are_inverses():
    grid = occupancy.decode_map(grid_header(width=20, height=20),
                                cells(20, 20).tobytes())
    col, row = grid.world_to_cell(np.array([-0.9]), np.array([-1.9]))
    x, y = grid.cell_to_world(col, row)
    # Back to within half a cell: cell_to_world returns centres.
    assert abs(x[0] - (-0.9)) <= grid.resolution
    assert abs(y[0] - (-1.9)) <= grid.resolution


def test_a_rotated_origin_is_honoured():
    """A SLAM node that starts its map aligned to the robot has a nonzero yaw.

    Ignoring it rotates the whole map by the robot's initial heading, which
    looks like a map and navigates like a mirror.
    """
    header = grid_header(width=10, height=10, origin=[0.0, 0.0, math.pi / 2])
    grid = occupancy.decode_map(header, cells(10, 10).tobytes())
    # One cell along the grid's own +x is one cell along world +y once the grid
    # is rotated a quarter turn.
    x, y = grid.cell_to_world(np.array([0]), np.array([0]))
    assert x[0] == pytest.approx(-0.025)
    assert y[0] == pytest.approx(0.025)


def test_out_of_bounds_points_are_reported_not_clamped():
    """Clamping would build a wall out of everything that was never in the map."""
    grid = occupancy.decode_map(grid_header(), cells().tobytes())
    col, row = grid.world_to_cell(np.array([1000.0]), np.array([1000.0]))
    assert not grid.in_bounds(col, row).any()


def test_classify_treats_unknown_as_its_own_class():
    data = cells()
    data[0, 0] = -1
    data[0, 1] = 0
    data[0, 2] = 100
    data[0, 3] = 50          # observed, between the thresholds
    grid = occupancy.decode_map(grid_header(), data.tobytes())
    occupied, free, unknown = grid.classify()
    assert unknown[0, 0] and free[0, 1] and occupied[0, 2]
    assert not (occupied[0, 3] or free[0, 3] or unknown[0, 3])
    assert grid.counts()["uncertain"] >= 1


def test_max_merge_adds_an_obstacle():
    base = _grid(np.zeros((4, 4), dtype=np.int8))
    patch = _grid(np.full((2, 2), 100, dtype=np.int8), x0=1, y0=1)
    changed = occupancy.apply_patch(base, patch)
    assert changed == 4
    assert base.cells[1, 1] == 100


def test_max_merge_never_clears_an_observed_cell():
    """The rule that makes the server safe to be wrong.

    Its evidence is a monocular reconstruction. When that errs it invents a
    surface, and an invented obstacle costs a detour; the opposite policy would
    let a missing surface clear a wall the scanner saw, which costs a collision.
    """
    base = _grid(np.full((4, 4), 100, dtype=np.int8))
    patch = _grid(np.zeros((4, 4), dtype=np.int8))
    assert occupancy.apply_patch(base, patch) == 0
    assert (base.cells == 100).all()


def test_max_merge_does_not_let_unknown_win_over_an_observed_cell():
    """-1 is "no opinion", not "probability -1".

    It would lose the numeric maximum by accident here, and stop doing so the
    moment unknown is spelled differently — so it is masked out explicitly.
    """
    base = _grid(np.zeros((2, 2), dtype=np.int8))
    patch = _grid(np.full((2, 2), -1, dtype=np.int8))
    assert occupancy.apply_patch(base, patch) == 0
    assert (base.cells == 0).all()


def test_max_merge_fills_in_unknown_cells():
    base = _grid(np.full((2, 2), -1, dtype=np.int8))
    patch = _grid(np.full((2, 2), 80, dtype=np.int8))
    assert occupancy.apply_patch(base, patch) == 4


def test_replace_merge_does_lower_a_cell():
    base = _grid(np.full((2, 2), 100, dtype=np.int8))
    patch = _grid(np.zeros((2, 2), dtype=np.int8))
    assert occupancy.apply_patch(base, patch, merge=wire.MAP_MERGE_REPLACE) == 4
    assert (base.cells == 0).all()


def test_a_patch_hanging_off_the_edge_is_clipped_not_dropped():
    """Normal when the robot's map has grown since the server's copy was taken."""
    base = _grid(np.zeros((4, 4), dtype=np.int8))
    patch = _grid(np.full((3, 3), 100, dtype=np.int8), x0=3, y0=3)
    assert occupancy.apply_patch(base, patch) == 1
    assert base.cells[3, 3] == 100


def test_a_patch_entirely_outside_changes_nothing():
    base = _grid(np.zeros((4, 4), dtype=np.int8))
    patch = _grid(np.full((2, 2), 100, dtype=np.int8), x0=50, y0=50)
    assert occupancy.apply_patch(base, patch) == 0


def test_patch_from_diff_returns_none_when_nothing_changed():
    """The common case, and the one worth making free."""
    base = _grid(np.zeros((8, 8), dtype=np.int8))
    assert occupancy.patch_from_diff(base, base.cells.copy()) is None


def test_patch_from_diff_is_the_bounding_box_of_the_changes():
    base = _grid(np.zeros((10, 10), dtype=np.int8))
    updated = base.cells.copy()
    updated[3, 4] = 100
    updated[5, 6] = 100
    patch = occupancy.patch_from_diff(base, updated)
    assert (patch.x0, patch.y0) == (4, 3)
    assert (patch.width, patch.height) == (3, 3)
    # Unchanged cells inside the box are unknown, so merging cannot touch them.
    assert patch.cells[0, 0] == 100
    assert patch.cells[1, 1] == wire.MAP_UNKNOWN


def test_a_patch_round_trips_through_apply():
    base = _grid(np.zeros((10, 10), dtype=np.int8))
    updated = base.cells.copy()
    updated[3, 4] = 100
    updated[5, 6] = 100
    patch = occupancy.patch_from_diff(base, updated)
    occupancy.apply_patch(base, patch)
    assert (base.cells == updated).all()


def test_analyse_reports_the_explored_fraction():
    data = np.full((10, 10), -1, dtype=np.int8)
    data[:5, :] = 0
    grid = _grid(data)
    assert occupancy.analyse(grid)["explored_fraction"] == pytest.approx(0.5)


def _grid(cells_array, **overrides):
    kwargs = {"resolution": 0.05, "origin": (0.0, 0.0, 0.0), "frame": "map",
              "map_id": "m1", "occupied_min": 65, "free_max": 25}
    kwargs.update(overrides)
    return Grid(cells=cells_array, **kwargs)
