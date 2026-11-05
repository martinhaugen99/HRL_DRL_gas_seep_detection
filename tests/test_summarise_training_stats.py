"""Tests for the training run summary in HUGIN_gym.evaluation.summarise_training_stats."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from HUGIN_gym.evaluation.summarise_training_stats import (
    ENDED_EARLY,
    ended_early_limit,
    episode_metrics,
    main,
)

LIMIT = 160


def make_config(option_lengths: list[int] | None) -> dict:
    """A training config; None leaves out HRL_VARIANT, as in the runs before run 5."""
    config: dict = {"ENVIRONMENT": {"max_episode_length": LIMIT}}
    if option_lengths is not None:
        config["HRL_VARIANT"] = {"option_lengths": option_lengths}
    return config


def make_stats(lengths: list[int]) -> dict:
    """Episode stats as EpisodeStatsCallback saves them, one entry per episode length."""
    return {
        "episode_rewards": [[-1.0] * length for length in lengths],
        "episode_lengths": lengths,
        "visited_states_counts": [0.5] * len(lengths),
        "episode_counter_above_threshold": [0.9] * len(lengths),
    }


@pytest.mark.parametrize(
    ("option_lengths", "expected"),
    [(None, LIMIT), ([3], LIMIT), ([2, 5], None)],
)
def test_ended_early_limit_is_none_only_with_several_option_lengths(
    option_lengths: list[int] | None, expected: int | None
) -> None:
    # Act
    limit = ended_early_limit(make_config(option_lengths))

    # Assert
    assert limit == expected


def test_episode_metrics_counts_shorter_episodes_as_ended_early() -> None:
    # Arrange
    stats = make_stats([LIMIT, LIMIT - 1, 100, LIMIT])

    # Act
    metrics = episode_metrics(stats, LIMIT, last=4)

    # Assert
    np.testing.assert_array_equal(metrics[ENDED_EARLY], [0, 100, 100, 0])


def test_episode_metrics_has_no_ended_early_values_without_a_limit() -> None:
    # Arrange: duration variant, where every episode is shorter than the limit
    stats = make_stats([100, 120])

    # Act
    metrics = episode_metrics(stats, None, last=2)

    # Assert
    assert metrics[ENDED_EARLY].size == 0
    assert metrics["Episode length"].size == 2


def test_main_prints_a_dash_for_ended_early_of_duration_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange: two repeats of a duration variant, so the table also has a group row
    run_dirs = [tmp_path / "HRL_PPO_duration_1", tmp_path / "HRL_PPO_duration_2"]
    for run_dir in run_dirs:
        run_dir.mkdir()
        with (run_dir / "training_stats.pkl").open("wb") as f:
            pickle.dump(make_stats([100, 120]), f)
        (run_dir / "training_config.json").write_text(json.dumps(make_config([2, 5])))
    monkeypatch.setattr(sys, "argv", ["summarise_training_stats", *map(str, run_dirs)])

    # Act
    main()

    # Assert
    rows = [line for line in capsys.readouterr().out.splitlines() if "HRL_PPO" in line]
    assert len(rows) == 3
    assert [row.rstrip(" |").rsplit("| ", 1)[1] for row in rows] == ["-", "-", "**-**"]
