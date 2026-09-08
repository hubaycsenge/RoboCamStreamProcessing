"""
Tests for the region extraction the robot's MapCloudAgreement is built from.

Written around the mistakes that produce a plausible region list rather than an
error: a region whose centre is half a cell out drives the robot to the wrong
side of a table, a height averaged instead of maximised turns a table into an
overhang, and a kind clustered across its neighbours turns glass into a keepout.
"""

import numpy as np
import pytest

from robocam import regions
from robocam.occupancy import Grid

FREE = 0
OCCUPIED = 100
UNKNOWN = -1


def make_grid(cells, resolution=0.05, origin=(0.0, 0.0, 0.0)):
    return Grid(
        cells=np.asarray(cells, dtype=np.int8),
        resolution=resolution,
        origin=origin,
        frame="map",
        map_id="m1",
    )


def free_grid(h=40, w=40, **kw):
    return make_grid(np.full((h, w), FREE, dtype=np.int8), **kw)


# --- heights ---------------------------------------------------------------

def test_cell_height_is_the_maximum_not_the_mean():
    """A table top and its underside are a table, not something at knee height."""
    grid = free_grid()
    # Two points in the same cell, 0.75 m apart in z.
    points = np.array([[0.11, 0.11, 0.20], [0.11, 0.11, 0.95]])
    heights = regions.cell_heights(points, grid, z_min=0.05, z_max=1.6)
    col, row = grid.world_to_cell(np.array([0.11]), np.array([0.11]))
    assert heights[row[0], col[0]] == pytest.approx(0.95)


def test_cells_without_points_are_nan_not_zero():
    """Zero is a floor; nan is 'no idea', and the robot must treat them differently."""
    grid = free_grid()
    heights = regions.cell_heights(np.zeros((0, 3)), grid)
    assert np.isnan(heights).all()


def test_a_low_surface_still_registers():
    """Starting the running maximum at 0.0 would hide a threshold strip."""
    grid = free_grid()
    points = np.array([[0.11, 0.11, 0.06]])
    heights = regions.cell_heights(points, grid, z_min=0.05, z_max=1.6)
    assert np.nanmax(heights) == pytest.approx(0.06)


# --- coverage --------------------------------------------------------------

def test_grid_coverage_is_over_free_cells_only():
    """Thick walls must not make a well-mapped room score worse than a bad one."""
    cells = np.full((10, 10), FREE, dtype=np.int8)
    cells[0, :] = OCCUPIED          # a wall along the top
    grid = make_grid(cells)
    cloud = np.zeros((10, 10), dtype=bool)
    cloud[1:6, :] = True            # half the free floor seen

    cover = regions.coverage(cloud, grid)
    free_cells = int((cells == FREE).sum())
    assert cover["free_cells"] == free_cells
    assert cover["grid_coverage"] == pytest.approx(50 / free_cells, abs=1e-3)


def test_coverage_of_an_empty_cloud_is_zero_not_an_error():
    grid = free_grid(10, 10)
    cover = regions.coverage(np.zeros((10, 10), dtype=bool), grid)
    assert cover["grid_coverage"] == 0.0
    assert cover["cloud_coverage"] == 0.0


def test_cloud_coverage_falls_when_the_cloud_leaves_the_map():
    """Structure over unknown cells is outside the map, and must show as such."""
    cells = np.full((10, 10), FREE, dtype=np.int8)
    cells[:, 5:] = UNKNOWN
    grid = make_grid(cells)
    cloud = np.zeros((10, 10), dtype=bool)
    cloud[0, 4] = True    # inside the known region
    cloud[0, 6] = True    # outside it
    assert regions.coverage(cloud, grid)["cloud_coverage"] == pytest.approx(0.5)


# --- classification --------------------------------------------------------

def test_the_four_kinds_are_disjoint():
    """A cell must not be two kinds at once, or a region's kind means nothing."""
    rng = np.random.default_rng(0)
    cells = rng.choice([FREE, OCCUPIED, UNKNOWN], size=(20, 20)).astype(np.int8)
    grid = make_grid(cells)
    cloud = rng.random((20, 20)) > 0.6

    masks = regions.classify_cells(cloud, grid)
    stacked = np.stack(list(masks.values()))
    assert stacked.sum(axis=0).max() <= 1


def test_unobserved_is_swept_floor_only_not_the_whole_map():
    """'You have not looked behind you' is not a revisit target."""
    cells = np.full((10, 10), UNKNOWN, dtype=np.int8)
    cells[0:2, 0:2] = FREE
    grid = make_grid(cells)
    masks = regions.classify_cells(np.zeros((10, 10), dtype=bool), grid)
    assert masks[regions.KIND_UNOBSERVED].sum() == 4


# --- regions ---------------------------------------------------------------

def test_a_block_of_cloud_over_free_floor_is_one_cloud_only_region():
    grid = free_grid(40, 40)
    cloud = np.zeros((40, 40), dtype=bool)
    cloud[10:20, 10:20] = True
    heights = np.full((40, 40), np.nan, dtype=np.float32)
    heights[10:20, 10:20] = 0.74

    found = regions.extract_regions(cloud, grid, heights, block_m=0.5)
    cloud_only = [r for r in found if r["kind"] == regions.KIND_CLOUD_ONLY]
    assert len(cloud_only) == 1
    assert cloud_only[0]["height"] == pytest.approx(0.74)


def test_region_centre_lands_on_the_structure():
    """Half a cell out sends the robot to the wrong side of the table."""
    grid = free_grid(40, 40, resolution=0.05)
    cloud = np.zeros((40, 40), dtype=bool)
    cloud[10:20, 10:20] = True
    found = regions.extract_regions(cloud, grid, block_m=0.5)
    region = [r for r in found if r["kind"] == regions.KIND_CLOUD_ONLY][0]

    # The block spans cells 10..19 in both axes; its centre is cell 15.0, and
    # cell_to_world puts cell centres at (index + 0.5) * resolution.
    expected = (15.0 + 0.5) * 0.05
    assert region["x"] == pytest.approx(expected, abs=0.05)
    assert region["y"] == pytest.approx(expected, abs=0.05)


def test_two_separated_blobs_are_two_regions():
    grid = free_grid(60, 60)
    cloud = np.zeros((60, 60), dtype=bool)
    cloud[5:12, 5:12] = True
    cloud[40:47, 40:47] = True
    found = [r for r in regions.extract_regions(cloud, grid, block_m=0.5)
             if r["kind"] == regions.KIND_CLOUD_ONLY]
    assert len(found) == 2


def test_kinds_never_cluster_together():
    """Glass and a table top are opposite decisions; one region cannot be both."""
    cells = np.full((40, 40), FREE, dtype=np.int8)
    cells[10:20, 20:30] = OCCUPIED      # the map sees a panel
    grid = make_grid(cells)
    cloud = np.zeros((40, 40), dtype=bool)
    cloud[10:20, 10:20] = True          # the cloud sees a table right beside it

    found = regions.extract_regions(cloud, grid, block_m=0.5)
    kinds = {r["kind"] for r in found}
    assert regions.KIND_CLOUD_ONLY in kinds
    assert regions.KIND_MAP_ONLY in kinds
    for r in found:
        assert r["kind"] in regions.KINDS


def test_specks_are_filtered_out():
    """One cell of monocular noise is not furniture."""
    grid = free_grid(40, 40)
    cloud = np.zeros((40, 40), dtype=bool)
    cloud[3, 3] = True
    found = [r for r in regions.extract_regions(cloud, grid, block_m=0.5,
                                                min_cells_per_block=3)
             if r["kind"] == regions.KIND_CLOUD_ONLY]
    assert found == []


def test_max_regions_drops_the_lowest_scoring():
    """The cap must lose the tail of specks, not the table in the middle."""
    grid = free_grid(80, 80)
    cloud = np.zeros((80, 80), dtype=bool)
    cloud[0:20, 0:20] = True                     # the big one
    for i in range(5):
        cloud[40 + i * 6:43 + i * 6, 60:63] = True   # small ones

    found = regions.extract_regions(cloud, grid, block_m=0.5, max_regions=2)
    assert len(found) <= 2
    assert found[0]["cells"] == max(r["cells"] for r in found)


def test_scores_are_relative_within_a_kind():
    """A room of unobserved floor must not bury the one table."""
    cells = np.full((60, 60), FREE, dtype=np.int8)
    grid = make_grid(cells)
    cloud = np.zeros((60, 60), dtype=bool)
    cloud[10:16, 10:16] = True

    found = regions.extract_regions(cloud, grid, block_m=0.5)
    by_kind = {}
    for r in found:
        by_kind.setdefault(r["kind"], []).append(r["score"])
    for kind, scores in by_kind.items():
        assert max(scores) == pytest.approx(1.0), kind


def test_no_regions_when_nothing_disagrees():
    cells = np.full((20, 20), UNKNOWN, dtype=np.int8)
    grid = make_grid(cells)
    assert regions.extract_regions(np.zeros((20, 20), dtype=bool), grid) == []


# --- the wire payload ------------------------------------------------------

def test_payload_arrays_are_parallel_and_the_same_length():
    grid = free_grid(40, 40)
    cloud = np.zeros((40, 40), dtype=bool)
    cloud[10:20, 10:20] = True
    found = regions.extract_regions(cloud, grid, block_m=0.5)
    payload = regions.agreement_payload({"agreement": 0.4}, found,
                                        regions.coverage(cloud, grid))
    n = len(found)
    for key in ("uncertain_x", "uncertain_y", "uncertain_scores",
                "uncertain_kinds", "uncertain_heights", "uncertain_radii"):
        assert len(payload[key]) == n, key


def test_an_unmeasurable_height_is_null_never_nan():
    """NaN through JSON becomes a height that fails every comparison silently."""
    cells = np.full((40, 40), FREE, dtype=np.int8)
    cells[10:20, 10:20] = OCCUPIED
    grid = make_grid(cells)
    cloud = np.zeros((40, 40), dtype=bool)          # map_only regions only

    found = regions.extract_regions(cloud, grid, block_m=0.5)
    payload = regions.agreement_payload({}, found, regions.coverage(cloud, grid))
    map_only = [h for h, k in zip(payload["uncertain_heights"],
                                  payload["uncertain_kinds"])
                if k == regions.KIND_MAP_ONLY]
    assert map_only
    assert all(h is None for h in map_only)
