"""Tests for the meta agent variants in HUGIN_gym/training/train_HRL(ppo).py and the job script that submits them."""

import importlib.util
import inspect
import re
from pathlib import Path
from types import ModuleType

import pytest

from HUGIN_gym.envs.HRL_env import HierarchicalEnv
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper

REPOSITORY = Path(__file__).parents[1]
SCRIPT = REPOSITORY / "HUGIN_gym" / "training" / "train_HRL(ppo).py"
MATRIX_SCRIPT = REPOSITORY / "fox_train_HRL_matrix.sh"
SURVEY_KEYS = {"spacing", "handover", "goal_mode", "goal_mean_std", "known_std", "bonus_mode", "sources"}


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """The training script as a module; its file name is not a valid module name."""
    spec = importlib.util.spec_from_file_location("train_HRL_ppo", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_preset_uses_only_settings_of_the_env_and_the_survey(script: ModuleType) -> None:
    # Arrange
    env_settings = set(inspect.signature(HierarchicalEnv.__init__).parameters)

    # Act and Assert
    for name, preset in script.PRESETS.items():
        assert set(preset) - {"survey"} <= env_settings, name
        assert set(preset.get("survey", {})) <= SURVEY_KEYS, name
        assert preset["meta_state"] in HierarchicalEnv.META_STATES, name
        assert preset.get("reward_mode", "cells") in HierarchicalEnv.REWARD_MODES, name
        assert preset.get("survey", {}).get("bonus_mode", "coverage") in GPWrapper.BONUS_MODES, name


def test_the_presets_of_runs_4_and_5_are_unchanged(script: ModuleType) -> None:
    # Act and Assert: they keep the reward, the options and the goal they ran with
    for name in ("untuned", "duration", "duration_mean", "minimal", "minimal_values", "minimal_values_duration"):
        assert "survey" not in script.PRESETS[name]
        assert "reward_mode" not in script.PRESETS[name]


def test_map_gain_has_no_coverage_bonus_on_top(script: ModuleType) -> None:
    # Act and Assert: it pays for a better map already
    assert script.PRESETS["sweep6_est_map"]["survey"]["bonus_mode"] == "none"


def test_the_agent_is_trained_on_a_random_number_of_plumes_except_the_comparison(script: ModuleType) -> None:
    # Arrange
    run_6 = [name for name, preset in script.PRESETS.items() if "survey" in preset]

    # Act and Assert: a fixed number would teach it how many to expect
    assert len(run_6) == 7
    for name in run_6:
        expected = [1, 1] if name == "sweep6_est_one" else [1, 3]
        assert script.PRESETS[name]["survey"]["sources"] == expected


def test_the_control_differs_from_the_main_variant_only_in_meta_state_reward_and_bonus(script: ModuleType) -> None:
    # Arrange
    main, control = script.PRESETS["sweep6_est"], script.PRESETS["sweep6_est_control"]

    # Act
    different_settings = {key for key in main if main[key] != control[key]}
    different_survey = {key for key in main["survey"] if main["survey"][key] != control["survey"][key]}

    # Assert: a difference in results then comes from these three fixes, with the same sweep, handover, goal and plumes
    assert set(main) == set(control) and set(main["survey"]) == set(control["survey"])
    assert different_settings == {"meta_state", "reward_mode", "survey"}
    assert different_survey == {"bonus_mode"}
    assert (control["meta_state"], control["reward_mode"], control["survey"]["bonus_mode"]) == (
        "minimal_values",
        "cells",
        "coverage",
    )


def test_the_matrix_script_submits_only_known_presets(script: ModuleType) -> None:
    # Act
    match = re.search(r"^TAGS=\"(.+)\"$", MATRIX_SCRIPT.read_text(), flags=re.MULTILINE)

    # Assert: all the variants of run 6
    assert match is not None
    assert set(match.group(1).split()) == {name for name, preset in script.PRESETS.items() if "survey" in preset}
