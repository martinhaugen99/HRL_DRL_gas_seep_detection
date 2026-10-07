"""Tests for saving and resuming the episode stats snapshots of EpisodeStatsCallback."""

import pickle
from pathlib import Path

import pytest

from HUGIN_gym.callbacks.EpisodeStatsCallback import EpisodeStatsCallback


def make_callback(tmp_path: Path) -> EpisodeStatsCallback:
    return EpisodeStatsCallback(
        save_freq=1, NUM_ENVS=2, address=str(tmp_path / "episode_stats"), GP=True
    )


def make_stats(episode_lengths: list[int], coverage: list[float]) -> dict:
    """Stats as get_stats returns them; the coverage lists skip some episodes."""
    return {
        "episode_rewards": [[-0.7] * length for length in episode_lengths],
        "episode_lengths": episode_lengths,
        "visited_states_counts": list(range(len(episode_lengths))),
        "episode_counter_above_threshold": coverage,
        "episode_counter_around_threshold": [],
        "assumed_episode_counter_above_threshold": coverage,
        "assumed_episode_counter_around_threshold": [],
    }


def write_snapshot(tmp_path: Path, step: int, stats: dict) -> Path:
    path = tmp_path / f"episode_stats_{step}.pkl"
    with path.open("wb") as f:
        pickle.dump(stats, f)
    return path


def test_load_snapshot_drops_episodes_after_checkpoint(tmp_path: Path) -> None:
    # Arrange: snapshot at step 80, checkpoint at step 60
    write_snapshot(tmp_path, 80, make_stats([20, 20, 20, 20], [0.1, 0.2, 0.3, 0.4]))
    callback = make_callback(tmp_path)

    # Act
    summary = callback.load_snapshot(up_to_step=60)

    # Assert
    stats = callback.get_stats()
    assert stats["episode_lengths"] == [20, 20, 20]
    assert len(stats["episode_rewards"]) == 3
    assert stats["visited_states_counts"] == [0, 1, 2]
    assert stats["episode_counter_above_threshold"] == [0.1, 0.2, 0.3]
    assert stats["assumed_episode_counter_above_threshold"] == [0.1, 0.2, 0.3]
    assert "episode_stats_80.pkl: kept 3 of 4 episodes" in summary


def test_load_snapshot_skips_unreadable_snapshot(tmp_path: Path) -> None:
    # Arrange: the newest snapshot was cut short by a failed write
    write_snapshot(tmp_path, 60, make_stats([20, 20, 20], [0.5]))
    broken = write_snapshot(tmp_path, 80, make_stats([20, 20, 20, 20], [0.5]))
    broken.write_bytes(broken.read_bytes()[:40])
    callback = make_callback(tmp_path)

    # Act
    summary = callback.load_snapshot(up_to_step=60)

    # Assert
    assert callback.get_stats()["episode_lengths"] == [20, 20, 20]
    assert "episode_stats_60.pkl" in summary
    assert "skipped unreadable episode_stats_80.pkl" in summary


def test_load_snapshot_raises_when_snapshots_are_older_than_checkpoint(
    tmp_path: Path,
) -> None:
    # Arrange
    write_snapshot(tmp_path, 40, make_stats([20, 20], [0.5]))
    callback = make_callback(tmp_path)

    # Act and assert
    with pytest.raises(FileNotFoundError):
        callback.load_snapshot(up_to_step=60)


@pytest.mark.parametrize(
    ("new_step", "kept_files"),
    [
        (100, {"episode_stats_100.pkl"}),  # the loaded snapshot is replaced
        (80, {"episode_stats_80.pkl"}),  # same name as the loaded snapshot: keep it
    ],
)
def test_save_snapshot_after_resume_keeps_only_newest(
    tmp_path: Path, new_step: int, kept_files: set[str]
) -> None:
    # Arrange: resume at step 60 from the snapshot at step 80
    write_snapshot(tmp_path, 80, make_stats([20, 20, 20, 20], [0.5, 0.5]))
    callback = make_callback(tmp_path)
    callback.load_snapshot(up_to_step=60)
    callback.num_timesteps = new_step

    # Act
    callback._save_snapshot()

    # Assert
    assert {path.name for path in tmp_path.iterdir()} == kept_files
    with (tmp_path / f"episode_stats_{new_step}.pkl").open("rb") as f:
        assert pickle.load(f)["episode_lengths"] == [20, 20, 20]
