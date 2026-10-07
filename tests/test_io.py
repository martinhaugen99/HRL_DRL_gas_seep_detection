"""Tests for the resume helpers in HUGIN_gym.utils.io."""

from pathlib import Path

import pytest

from HUGIN_gym.utils.io import append_resume_log, find_latest_checkpoint


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
