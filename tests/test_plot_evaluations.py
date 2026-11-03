"""Tests for HUGIN_gym/evaluation/plot_evaluations.py: the calculations, and that each figure can be saved."""

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")

from HUGIN_gym.evaluation import plot_evaluations as plots  # after the backend is set
from HUGIN_gym.evaluation.summarise_evaluations import Result


def result(
    survey: str, actions: str, seed: int | None, steps: float = 100.0, plumes: int = 1
) -> Result:
    episode = {
        "steps": steps,
        "ended_early": 1.0,
        "plume_overlap": 0.99,
        "border_overlap": 0.98,
        "true_plume_visited": 0.9,
        "true_border_visited": 0.8,
        "plumes": plumes,
        "plumes_found": plumes,
        "plumes_visited": plumes,
        "turns": 10,
        "decisions": 3,
        "reward": -5.0,
        "rmse": 0.01,
        "plume_key": "a",
        "actions": [0, 1, 2, 0],
    }
    return Result(
        "name",
        "sources_1_3",
        survey,
        actions,
        seed,
        {},
        ["sweep_3", "plume_3", "border_3"],
        [episode],
    )


def write_run(agents_dir: Path, name: str, values: list[float]) -> Path:
    run_dir = agents_dir / name
    writer = SummaryWriter(str(run_dir / "tensorboard" / "PPO_1"))
    for index, value in enumerate(values):
        writer.add_scalar("hrl/episode_steps", value, (index + 1) * 6144)
        writer.add_scalar("hrl/ended_early", 0.9, (index + 1) * 6144)
    writer.close()
    return run_dir


def test_smooth_keeps_the_length_and_a_constant_series() -> None:
    # Act
    smoothed = plots.smooth(np.full(40, 3.0))

    # Assert
    assert smoothed == pytest.approx(np.full(40, 3.0))


def test_smooth_uses_a_shorter_window_at_the_start() -> None:
    # Act
    smoothed = plots.smooth(np.array([2.0, 4.0, 6.0]), window=2)

    # Assert
    assert smoothed == pytest.approx([2.0, 3.0, 5.0])


def test_read_curve_reads_the_decisions_and_the_values(tmp_path: Path) -> None:
    # Arrange
    run_dir = write_run(tmp_path, "HRL_PPO_sweep6_est_seed1_1", [10.0, 20.0])

    # Act
    steps, values = plots.read_curve(run_dir, "hrl/episode_steps")

    # Assert
    assert steps == pytest.approx([6144, 12288])
    assert values == pytest.approx([10.0, 20.0])


def test_read_curve_tells_when_a_run_has_no_events(tmp_path: Path) -> None:
    # Act and assert
    with pytest.raises(FileNotFoundError):
        plots.read_curve(tmp_path, "hrl/episode_steps")


def test_read_curve_tells_when_a_run_did_not_log_the_value(tmp_path: Path) -> None:
    # Arrange: runs before run 5 have no hrl/ values, and must not vanish from a figure in silence
    run_dir = write_run(tmp_path, "HRL_PPO_untuned_1", [10.0])

    # Act and assert
    with pytest.raises(ValueError, match="hrl/not_logged"):
        plots.read_curve(run_dir, "hrl/not_logged")


def test_plot_learning_curves_tells_when_a_variant_has_no_runs(tmp_path: Path) -> None:
    # Act and assert
    with pytest.raises(FileNotFoundError, match="sweep6_est"):
        plots.plot_learning_curves(tmp_path, [])


def test_seed_band_gives_the_mean_and_the_outer_seeds() -> None:
    # Arrange
    steps = np.arange(40, dtype=float)

    # Act
    grid, mean, low, high = plots.seed_band(
        [(steps, np.full(40, 10.0)), (steps, np.full(40, 20.0))]
    )

    # Assert
    assert grid == pytest.approx(steps)
    assert mean == pytest.approx(np.full(40, 15.0))
    assert low == pytest.approx(np.full(40, 10.0))
    assert high == pytest.approx(np.full(40, 20.0))


def assert_saved(fig: matplotlib.figure.Figure, folder: Path, name: str) -> None:
    plots.save(fig, folder, name)
    assert (folder / f"{name}.pdf").stat().st_size > 0
    assert (folder / f"{name}.png").stat().st_size > 0


def test_plot_surveys_saves_a_figure_with_a_row_per_survey(tmp_path: Path) -> None:
    # Arrange
    results = [
        result("sweep6_est", "sampled", 1, 235),
        result("sweep6_est", "deterministic", 1, 252),
        result("rule_sweep6_handover_estimate", "scripted", None, 231),
    ]

    # Act
    fig = plots.plot_surveys(results)

    # Assert
    assert len(fig.axes[0].get_yticks()) == 2
    assert_saved(fig, tmp_path, "surveys")


def test_plot_choices_saves_a_figure_with_a_panel_per_number_of_plumes(
    tmp_path: Path,
) -> None:
    # Arrange
    results = [
        result("sweep6_est", "sampled", 1, plumes=2),
        result("rule_sweep6_handover_estimate", "scripted", None, plumes=1),
    ]

    # Act
    fig = plots.plot_choices(results)

    # Assert
    assert len(fig.axes) == 6
    assert_saved(fig, tmp_path, "choices")


def test_plot_learning_curves_saves_a_figure_from_the_tensorboard_logs(
    tmp_path: Path,
) -> None:
    # Arrange
    agents_dir = tmp_path / "agents"
    for variant in plots.LEARNING_VARIANTS:
        for seed in (1, 2):
            write_run(
                agents_dir, f"HRL_PPO_{variant}_seed{seed}_1", [300.0, 250.0, 240.0]
            )
    results = [result("lawnmower_sweep3_estimate", "scripted", None, 282)]

    # Act
    fig = plots.plot_learning_curves(agents_dir, results)

    # Assert
    assert len(fig.axes) == 2
    assert_saved(fig, tmp_path, "learning_curves")
