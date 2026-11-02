"""Tests for the comparison of the true map and the estimated map."""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from HUGIN_gym.evaluation.compare_maps import (
    CORRECT,
    EXTRA,
    MISSED,
    NOT_GAS,
    EpisodeMaps,
    comparison_classes,
    parse_args,
    pick_episodes,
    plot_comparison,
    save_results,
)
from HUGIN_gym.evaluation.survey_metrics import map_metrics

matplotlib.use("Agg")  # no plot windows in the tests

SIZE = 41
THRESHOLD, WIDTH = 0.3, 0.15


def make_episode(offset: float = 0.0, ended: bool = True) -> EpisodeMaps:
    """A Gaussian plume in the middle of the map, with an estimate shifted by offset cells."""
    coords = np.arange(-20, 21)
    x, y = np.meshgrid(coords, coords, indexing="ij")
    true_concentration = np.exp(-(x**2 + y**2) / (2 * 3.5**2))
    gp_mean = np.exp(-((x - offset) ** 2 + y**2) / (2 * 3.5**2))
    true_plume = true_concentration >= THRESHOLD
    true_border = (true_concentration <= THRESHOLD + WIDTH) & (true_concentration >= THRESHOLD - WIDTH)
    visited = np.abs(y) < 5
    results = map_metrics(gp_mean, true_plume, true_border, visited, THRESHOLD, WIDTH)
    results.update(ended_early=float(ended), steps=120.0, visited_share=float(np.mean(visited)), mean_gp_std=0.2)
    path = np.array([[-19.0, 0.0], [-17.0, 0.0], [-15.0, 0.0], [-13.0, 0.0]])
    return EpisodeMaps(
        true_concentration,
        gp_mean,
        1 - np.exp(-(x**2) / 50),
        true_plume,
        true_border,
        visited,
        path,
        np.array([0, 1, 2]),
        THRESHOLD,
        WIDTH,
        results,
    )


def test_comparison_classes_marks_correct_missed_and_extra_cells() -> None:
    # Arrange
    true_map = np.array([[True, True], [False, False]])
    estimated_map = np.array([[True, False], [True, False]])

    # Act
    classes = comparison_classes(true_map, estimated_map)

    # Assert
    np.testing.assert_array_equal(classes, [[CORRECT, MISSED], [EXTRA, NOT_GAS]])


def test_pick_episodes_takes_the_worst_the_median_and_the_best() -> None:
    # Arrange: scores 0.0 to 1.0 in a mixed order
    scores = [0.5, 0.0, 1.0, 0.25, 0.75, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7]

    # Act
    picked = dict(pick_episodes(scores))

    # Assert
    assert scores[picked["Worst"]] == 0.0
    assert scores[picked["10th percentile"]] == 0.1
    assert scores[picked["Median"]] == 0.4
    assert scores[picked["Best"]] == 1.0


@pytest.mark.parametrize(("scores", "expected"), [([0.5], 1), ([0.2, 0.9], 2)])
def test_pick_episodes_does_not_show_an_episode_twice(scores: list[float], expected: int) -> None:
    # Act
    picked = pick_episodes(scores)

    # Assert
    assert len(picked) == expected
    assert len({index for _, index in picked}) == expected


def test_score_is_high_for_a_good_estimate() -> None:
    # Act and Assert
    assert make_episode(offset=0.0).score == 1.0
    assert make_episode(offset=4.0).score < 0.5


@pytest.mark.parametrize("rows", [1, 4])
def test_plot_comparison_saves_png_and_svg(tmp_path: Path, rows: int) -> None:
    # Arrange
    episodes = [(f"Episode {i}", make_episode(offset=float(i), ended=i == 0)) for i in range(rows)]

    # Act
    png_path = plot_comparison(episodes, "Sweep", tmp_path / "new_folder")

    # Assert
    assert png_path == tmp_path / "new_folder" / "map_comparison.png"
    assert png_path.stat().st_size > 0
    assert png_path.with_suffix(".svg").stat().st_size > 0


def test_plot_comparison_handles_an_estimate_with_no_gas(tmp_path: Path) -> None:
    # Arrange: no contour level is reached in the estimate
    episode = make_episode()
    episode.gp_mean = np.zeros((SIZE, SIZE))

    # Act
    png_path = plot_comparison([("Empty", episode)], "Space agent", tmp_path)

    # Assert
    assert png_path.exists()


def test_save_results_writes_one_row_per_episode(tmp_path: Path) -> None:
    # Act
    csv_path = save_results([make_episode(), make_episode(offset=2.0)], tmp_path)

    # Assert
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("episode,plume_overlap,border_overlap")
    assert len(lines) == 3
    assert lines[1].startswith("0,1.0000")


def test_parse_args_needs_an_output_folder(tmp_path: Path) -> None:
    # Act and Assert
    with pytest.raises(SystemExit):
        parse_args([])
    assert parse_args(["--output", str(tmp_path)]).output == tmp_path


@pytest.mark.parametrize("mode", ["2", "3", "5"])
def test_parse_args_rejects_modes_without_a_plot(tmp_path: Path, mode: str) -> None:
    # Act and Assert
    with pytest.raises(SystemExit):
        parse_args(["--output", str(tmp_path), "--mode", mode])
