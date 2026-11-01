"""Tests for the map metrics that compare trained agents and baselines."""

import numpy as np
import pytest

from HUGIN_gym.evaluation.survey_metrics import map_metrics, overlap, share, summarise_by_plumes


def empty_map() -> np.ndarray:
    return np.zeros((10, 10), dtype=bool)


def test_overlap_of_identical_and_of_partly_equal_maps() -> None:
    # Arrange
    a, b = empty_map(), empty_map()
    a[0:2, 0:2] = True  # 4 cells
    b[0:2, 0:3] = True  # 6 cells, 4 in common

    # Act and Assert
    assert overlap(a, a) == 1.0
    assert overlap(a, b) == pytest.approx(4 / 6)
    assert overlap(empty_map(), empty_map()) == 1.0


def test_share_of_an_empty_map_is_complete() -> None:
    # Arrange
    whole = empty_map()
    whole[0, 0:4] = True
    part = empty_map()
    part[0, 0] = True

    # Act and Assert
    assert share(part, whole) == pytest.approx(1 / 4)
    assert share(part, empty_map()) == 1.0


def test_map_metrics_counts_found_and_visited_plumes() -> None:
    # Arrange: two true plumes. The GP finds both, and the AUV visits only the first.
    true_over = empty_map()
    true_over[0:2, 0:2] = True
    true_over[6:8, 6:8] = True
    true_around = empty_map()
    true_around[2, 0:2] = True
    gp_mean = np.zeros((10, 10))
    gp_mean[true_over] = 0.9
    gp_mean[true_around] = 0.3  # the border band, 0.15 to 0.45
    visited = empty_map()
    visited[0:2, 0:2] = True

    # Act
    metrics = map_metrics(gp_mean, true_over, true_around, visited, c_threshold=0.3, c_width=0.15)

    # Assert
    assert metrics["plumes"] == 2
    assert metrics["plumes_found"] == 2
    assert metrics["plumes_visited"] == 1
    assert metrics["plume_overlap"] == 1.0
    assert metrics["border_overlap"] == 1.0
    assert metrics["true_plume_visited"] == pytest.approx(0.5)
    assert metrics["true_border_visited"] == 0.0


def test_map_metrics_counts_a_plume_the_gp_missed() -> None:
    # Arrange
    true_over = empty_map()
    true_over[0:2, 0:2] = True
    true_over[6:8, 6:8] = True
    gp_mean = np.zeros((10, 10))
    gp_mean[0:2, 0:2] = 0.9

    # Act
    metrics = map_metrics(gp_mean, true_over, empty_map(), empty_map(), c_threshold=0.3, c_width=0.15)

    # Assert
    assert metrics["plumes_found"] == 1
    assert metrics["plume_overlap"] == pytest.approx(0.5)


def test_overlapping_sources_count_as_one_plume() -> None:
    # Arrange: one connected region
    true_over = empty_map()
    true_over[0:2, 0:6] = True

    # Act
    metrics = map_metrics(np.zeros((10, 10)), true_over, empty_map(), empty_map(), c_threshold=0.3, c_width=0.15)

    # Assert
    assert metrics["plumes"] == 1


def test_summarise_by_plumes_groups_the_episodes() -> None:
    # Arrange
    rows = [
        {"plumes": 1, "steps": 100.0},
        {"plumes": 2, "steps": 200.0},
        {"plumes": 1, "steps": 300.0},
    ]

    # Act
    summary = summarise_by_plumes(rows)

    # Assert
    assert summary == [
        {"plumes": 1, "steps": 200.0, "episodes": 2},
        {"plumes": 2, "steps": 200.0, "episodes": 1},
    ]
