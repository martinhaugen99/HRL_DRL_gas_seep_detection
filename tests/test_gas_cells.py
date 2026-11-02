"""Tests for the helpers for the estimated gas maps."""

import numpy as np
import pytest

from HUGIN_gym.envs.core.gas_cells import gp_maps, nearest_distance, overlap, patch_mask, unfinished_cells


def empty_map() -> np.ndarray:
    return np.zeros((41, 41), dtype=bool)


def test_gp_maps_split_the_concentration_at_the_thresholds() -> None:
    # Arrange
    mean = np.array([0.1, 0.2, 0.35, 0.44, 0.5])

    # Act
    plume, border = gp_maps(mean, c_threshold=0.3, c_width=0.15)

    # Assert
    np.testing.assert_array_equal(plume, [False, False, True, True, True])
    np.testing.assert_array_equal(border, [False, True, True, True, False])


def test_overlap_of_identical_partly_equal_and_empty_maps() -> None:
    # Arrange
    a, b = empty_map(), empty_map()
    a[0:2, 0:2] = True  # 4 cells
    b[0:2, 0:3] = True  # 6 cells, 4 in common

    # Act and Assert
    assert overlap(a, a) == 1.0
    assert overlap(a, b) == pytest.approx(4 / 6)
    assert overlap(empty_map(), empty_map()) == 1.0


def test_unfinished_cells_follow_the_end_goal() -> None:
    # Arrange: three gas cells: one visited, one well known, and the corner cell, which is not gas
    belief, corners, visited = empty_map(), empty_map(), empty_map()
    belief[0, 0] = belief[5, 5] = belief[6, 6] = belief[7, 7] = True
    corners[0, 0] = True
    visited[5, 5] = True
    std = np.full((41, 41), 0.5)
    std[6, 6] = 0.05

    # Act
    for_visit = unfinished_cells(belief, corners, visited, std, "visit", 0.1)
    for_estimate = unfinished_cells(belief, corners, visited, std, "estimate", 0.1)

    # Assert
    assert sorted(zip(*np.nonzero(for_visit), strict=True)) == [(6, 6), (7, 7)]
    assert sorted(zip(*np.nonzero(for_estimate), strict=True)) == [(5, 5), (7, 7)]


def test_patch_mask_is_the_square_around_the_cell() -> None:
    # Act
    middle = patch_mask((41, 41), 20, 20, 3)
    corner = patch_mask((41, 41), 0, 0, 3)

    # Assert: 7 x 7 cells, cut off by the edge of the map
    assert middle.sum() == 49
    assert middle[17, 17] and middle[23, 23] and not middle[16, 20]
    assert corner.sum() == 16


def test_nearest_distance() -> None:
    # Arrange
    cells = empty_map()
    cells[10, 14] = True
    cells[30, 30] = True

    # Act and Assert
    assert nearest_distance(cells, 10, 10) == 4.0
    assert nearest_distance(empty_map(), 10, 10) == float("inf")
