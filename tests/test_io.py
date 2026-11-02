"""Tests for the resume helpers in HUGIN_gym.utils.io."""

from pathlib import Path

import pytest

from HUGIN_gym.utils.io import (
    append_resume_log,
    find_latest_checkpoint,
    hrl_survey_from_config,
    hrl_variant_from_config,
)

DEFAULT_VARIANT = {"option_lengths": [3], "smdp_discount": False, "meta_state": "full"}


def test_find_latest_checkpoint_compares_step_counts_as_numbers(tmp_path: Path) -> None:
    # Arrange: "9999984" sorts after "59999904" as text
    for name in [
        "DQN_checkpoint_9999984_steps.zip",
        "DQN_checkpoint_59999904_steps.zip",
        "DQN_checkpoint_replay_buffer_69999888_steps.pkl",
        "DQN_scratch.zip",
        "SAC_checkpoint_79999872_steps.zip",
    ]:
        (tmp_path / name).touch()

    # Act
    path, step = find_latest_checkpoint(tmp_path, "DQN_checkpoint")

    # Assert
    assert path == tmp_path / "DQN_checkpoint_59999904_steps.zip"
    assert step == 59_999_904


def test_find_latest_checkpoint_raises_without_checkpoints(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "DQN_scratch.zip").touch()

    # Act and assert
    with pytest.raises(FileNotFoundError):
        find_latest_checkpoint(tmp_path, "DQN_checkpoint")


def test_append_resume_log_appends_lines_with_slurm_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "4259607")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")

    # Act
    append_resume_log(tmp_path, "first")
    append_resume_log(tmp_path, "second")

    # Assert
    lines = (tmp_path / "resume_log.txt").read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("job 4259607_0: first")
    assert lines[1].endswith("job 4259607_0: second")


def test_hrl_variant_from_config_reads_hrl_variant() -> None:
    # Arrange: as written by build_training_config_ppo from run 5 on
    config = {
        "HRL_VARIANT": {
            "preset": "duration",
            "survey_steps": 480,
            "option_lengths": [2, 5],
            "smdp_discount": True,
            "meta_state": "full",
        }
    }

    # Act
    variant, survey_steps = hrl_variant_from_config(config, DEFAULT_VARIANT, 300)

    # Assert: only the HierarchicalEnv settings are left in the variant
    assert variant == {"option_lengths": [2, 5], "smdp_discount": True, "meta_state": "full"}
    assert survey_steps == 480
    assert "preset" in config["HRL_VARIANT"]  # the config is not changed


def test_hrl_variant_from_config_uses_defaults_without_hrl_variant() -> None:
    # Arrange: runs started before the config saved HRL_VARIANT (run 4)
    config = {"ENVIRONMENT": {"max_episode_length": 160}}

    # Act
    variant, survey_steps = hrl_variant_from_config(config, DEFAULT_VARIANT, 480)

    # Assert
    assert variant == DEFAULT_VARIANT
    assert variant is not DEFAULT_VARIANT
    assert survey_steps == 480


def test_hrl_variant_from_config_leaves_out_the_survey_and_the_seed() -> None:
    # Arrange: as written from run 6 on
    config = {
        "HRL_VARIANT": {
            "preset": "sweep6_est",
            "survey_steps": 480,
            "seed": 2,
            "option_lengths": [3],
            "reward_mode": "cells_fixed",
            "survey": {"spacing": 6},
        }
    }

    # Act
    variant, _ = hrl_variant_from_config(config, DEFAULT_VARIANT, 480)

    # Assert: only the HierarchicalEnv settings are left
    assert variant == {"option_lengths": [3], "reward_mode": "cells_fixed"}


def test_hrl_survey_from_config_reads_the_survey_of_the_run() -> None:
    # Arrange
    config = {"HRL_VARIANT": {"survey_steps": 480, "survey": {"spacing": 6, "goal_mode": "estimate"}}}

    # Act and Assert
    assert hrl_survey_from_config(config) == {"spacing": 6, "goal_mode": "estimate"}


@pytest.mark.parametrize("config", [{}, {"HRL_VARIANT": {"survey_steps": 480}}])
def test_hrl_survey_from_config_is_empty_for_runs_before_run_6(config: dict) -> None:
    # Act and Assert
    assert hrl_survey_from_config(config) == {}
