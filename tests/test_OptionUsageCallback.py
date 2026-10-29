"""Tests for OptionUsageCallback: the meta agent's choices per survey phase, option lengths and episode ends."""

from types import SimpleNamespace

import numpy as np
import pytest

from HUGIN_gym.callbacks.OptionUsageCallback import OptionUsageCallback


class FakeLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.records[key] = value


def info(survey_fraction: float, option_length: int, survey_steps: int, **extra: object) -> dict:
    return {"survey_fraction": survey_fraction, "option_length": option_length, "survey_steps": survey_steps, **extra}


def run_rollout(callback: OptionUsageCallback, steps: list[dict]) -> FakeLogger:
    logger = FakeLogger()
    callback.model = SimpleNamespace(logger=logger)
    for step in steps:
        callback.update_locals(step)
        callback._on_step()
    callback._on_rollout_end()
    return logger


def test_choices_are_shared_out_per_survey_phase() -> None:
    # Arrange: 2 envs, 2 decisions each
    callback = OptionUsageCallback(["space_2", "space_5"])
    steps = [
        {
            "actions": np.array([0, 1]),
            "infos": [info(0.0, 2, 2), info(0.1, 5, 55)],
            "dones": np.array([False, False]),
        },
        {
            "actions": np.array([1, 1]),
            "infos": [info(0.5, 5, 7), info(0.99, 5, 480, **{"TimeLimit.truncated": True})],
            "dones": np.array([False, True]),
        },
    ]

    # Act
    records = run_rollout(callback, steps).records

    # Assert
    assert records["hrl_choice_early/space_2"] == pytest.approx(0.5)
    assert records["hrl_choice_early/space_5"] == pytest.approx(0.5)
    assert records["hrl_choice_middle/space_5"] == pytest.approx(1.0)
    assert records["hrl_choice_late/space_5"] == pytest.approx(1.0)
    assert records["hrl/option_length"] == pytest.approx((2 + 5 + 5 + 5) / 4)
    assert records["hrl/episode_steps"] == 480
    assert records["hrl/ended_early"] == 0.0


def test_episode_that_reaches_the_goal_ended_early() -> None:
    # Arrange
    callback = OptionUsageCallback(["space_3"])
    steps = [{"actions": np.array([0]), "infos": [info(0.4, 3, 300)], "dones": np.array([True])}]

    # Act
    records = run_rollout(callback, steps).records

    # Assert
    assert records["hrl/ended_early"] == 1.0
    assert records["hrl/episode_steps"] == 300


def test_counts_start_again_after_each_rollout() -> None:
    # Arrange
    callback = OptionUsageCallback(["space_2", "space_5"])
    run_rollout(callback, [{"actions": np.array([0]), "infos": [info(0.0, 2, 2)], "dones": np.array([False])}])

    # Act
    records = run_rollout(
        callback, [{"actions": np.array([1]), "infos": [info(0.0, 5, 5)], "dones": np.array([False])}]
    ).records

    # Assert
    assert records["hrl_choice_early/space_5"] == 1.0
    assert "hrl/episode_steps" not in records
