"""Tests for the log of what the plume and border agents see and do inside a survey."""

from pathlib import Path

import numpy as np
import pytest

from HUGIN_gym.agents.options import GasFillOption, SweepOption
from HUGIN_gym.envs.HRL_env import HierarchicalEnv
from HUGIN_gym.envs.HUGIN_env import HUGIN
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
from HUGIN_gym.evaluation.baselines import make_baseline
from HUGIN_gym.evaluation.log_gas_agents import (
    nearest_distance,
    parse_args,
    patch_mask,
    run_episode,
    save_rows,
    summarise,
    unfinished_cells,
)


def empty_map() -> np.ndarray:
    return np.zeros((41, 41), dtype=bool)


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


def make_row(**values: object) -> dict:
    row = {
        "episode": 0,
        "option": "border",
        "survey_step": 30,
        "space_coverage": 0.2,
        "own_coverage": 0.5,
        "unfinished_total": 10,
        "unfinished_in_patch": 3,
        "gas_in_patch": 5,
        "nearest_unfinished": 2.0,
        "visited_in_patch": 0.4,
        "actions": "forward left forward",
        "steps": 3,
        "edge_hits": 0,
        "new_cells": 6,
        "target_cells_finished": 2,
        "idle": False,
        "repeated_pose": False,
        "episode_ended_by_goal": True,
    }
    return {**row, **values}


def test_summarise_separates_what_the_agent_could_see() -> None:
    # Arrange: useful with a target in the patch, idle without one
    rows = [
        make_row(),
        make_row(
            unfinished_in_patch=0,
            gas_in_patch=0,
            nearest_unfinished=8.0,
            target_cells_finished=0,
            idle=True,
            new_cells=0,
        ),
    ]

    # Act
    summary = summarise(rows)

    # Assert
    assert "border agent:      2 decisions | finished a target cell  50.0 % | idle  50.0 %" in summary
    assert "target cell in its patch: share  50.0 % |      1 decisions | finished a target cell 100.0 %" in summary
    assert (
        "no gas of its own in patch: share  50.0 % |      1 decisions | finished a target cell   0.0 % | idle 100.0 %"
        in summary
    )
    assert "6 to 10 cells: share  50.0 %" in summary
    assert "ended by the goal, 1 episodes" in summary


def test_save_rows_writes_one_line_per_decision(tmp_path: Path) -> None:
    # Act
    save_rows([make_row(), make_row(episode=1)], tmp_path / "new_folder" / "log.csv")

    # Assert
    lines = (tmp_path / "new_folder" / "log.csv").read_text().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("episode,option,survey_step")


def test_parse_args_rejects_settings_without_the_logged_agents() -> None:
    # Act and Assert
    with pytest.raises(SystemExit):
        parse_args(["--scripted-gas"])
    with pytest.raises(SystemExit):
        parse_args(["--mode", "5"])
    assert parse_args([]).output is None


def test_run_episode_logs_only_the_gas_decisions_and_restores_the_env() -> None:
    # Arrange: the real env and the rule, with scripted options instead of the trained models
    base = HUGIN()
    base.train = True
    base.use_c_map = True
    base.GP_ON = True
    base.HRL = True
    base.agent_type = "META"  # type: ignore[attr-defined]  # set by the scripts, not in HUGIN.__init__
    base.spawn_at_survey_start = True
    base.vary_plume_size = True
    gp_env = GPWrapper(base, HRL=True)
    original_step = gp_env.step
    options = [SweepOption(base, gp_env, spacing=4), GasFillOption(base, gp_env), GasFillOption(base, gp_env)]
    env = HierarchicalEnv(
        gp_env, options, gamma=0.998, MAX_OPT_LENGTH=3, ACCURACY_GOALS=[0.9, 0.9, 0.9], GP=True, max_primitive_steps=480
    )
    np.random.seed(1)

    # Act
    rows = run_episode(env, make_baseline("rule", env, seed=0), deterministic=True, episode=7)

    # Assert
    assert rows
    assert {row["option"] for row in rows} <= {"plume", "border"}
    assert all(row["episode"] == 7 and row["steps"] == 3 and row["episode_ended_by_goal"] for row in rows)
    assert any(row["target_cells_finished"] > 0 for row in rows)
    assert gp_env.step == original_step
