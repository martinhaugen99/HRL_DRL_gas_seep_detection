"""Tests for the training stats plot in HUGIN_gym.evaluation.training_data_visualiser."""

import pickle
import sys
from pathlib import Path

import matplotlib
import pytest

from HUGIN_gym.evaluation.training_data_visualiser import main, plot_training_stats

matplotlib.use("Agg")  # no plot windows in the tests

N_EPISODES = 20


def write_stats(run_dir: Path, with_coverage: bool = True) -> None:
    """Write a small training_stats.pkl, as EpisodeStatsCallback saves it, to run_dir."""
    stats = {
        "episode_rewards": [[-0.5, -0.5, 40.0]] * N_EPISODES,
        "episode_lengths": [3] * N_EPISODES,
        "visited_states_counts": [0.5] * N_EPISODES,
    }
    if with_coverage:
        stats["episode_counter_around_threshold"] = [0.8] * N_EPISODES
        stats["episode_counter_above_threshold"] = [0.9] * N_EPISODES
    with (run_dir / "training_stats.pkl").open("wb") as f:
        pickle.dump(stats, f)


@pytest.mark.parametrize("with_coverage", [True, False])
def test_plot_training_stats_saves_png_and_svg_in_the_run_folder(
    tmp_path: Path, with_coverage: bool
) -> None:
    # Arrange: the space agent's stats have no plume and border coverage
    write_stats(tmp_path, with_coverage)

    # Act
    png_path = plot_training_stats(tmp_path)

    # Assert
    assert png_path == tmp_path / "training_stats.png"
    assert png_path.stat().st_size > 0
    assert (tmp_path / "training_stats.svg").stat().st_size > 0


def test_main_plots_every_run_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    run_dirs = [tmp_path / "HRL_PPO_minimal_1", tmp_path / "HRL_PPO_duration_2"]
    for run_dir in run_dirs:
        run_dir.mkdir()
        write_stats(run_dir)
    monkeypatch.setattr(sys, "argv", ["training_data_visualiser", *map(str, run_dirs)])

    # Act
    main()

    # Assert
    for run_dir in run_dirs:
        assert (run_dir / "training_stats.png").exists()
        assert (run_dir / "training_stats.svg").exists()
