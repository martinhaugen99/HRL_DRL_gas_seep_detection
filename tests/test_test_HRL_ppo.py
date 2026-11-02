"""Tests for the command line of HUGIN_gym/evaluation/test_HRL(ppo).py."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from HUGIN_gym.agents.options import GasFillOption, SweepOption
from HUGIN_gym.envs.HRL_env import HierarchicalEnv
from HUGIN_gym.envs.HUGIN_env import HUGIN
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
from HUGIN_gym.evaluation.baselines import make_baseline

SCRIPT = Path(__file__).parents[1] / "HUGIN_gym" / "evaluation" / "test_HRL(ppo).py"
MODEL = Path("../trained-agents/HRL_PPO_minimal_4371279/HRL_PPO_scratch.zip")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """The evaluation script as a module; its file name is not a valid module name."""
    spec = importlib.util.spec_from_file_location("test_HRL_ppo", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_run_4_and_asks_for_the_mode(script: ModuleType) -> None:
    # Act
    args = script.parse_args([])

    # Assert
    assert args.model == script.DEFAULT_MODEL
    assert args.mode is None


@pytest.mark.parametrize("mode", ["1", "2", "3", "4", "5"])
def test_parse_args_reads_the_model_and_the_mode(script: ModuleType, mode: str) -> None:
    # Act
    args = script.parse_args(["--model", str(MODEL), "--mode", mode])

    # Assert
    assert args.model == MODEL
    assert args.mode == mode


def test_parse_args_rejects_an_unknown_mode(script: ModuleType) -> None:
    # Act and assert
    with pytest.raises(SystemExit):
        script.parse_args(["--mode", "6"])


def test_parse_args_defaults_keep_todays_evaluation(script: ModuleType) -> None:
    # Act
    args = script.parse_args([])

    # Assert
    assert args.baseline is None
    assert args.spacing is None
    assert args.sources is None
    assert args.goal == "visit"
    assert args.episodes == 1000


def test_parse_args_reads_a_baseline_survey(script: ModuleType) -> None:
    # Act
    args = script.parse_args(
        ["--baseline", "rule", "--spacing", "6", "--sources", "1", "3", "--goal", "estimate", "--goal-mean-std", "0.26", "--known-std", "0.15", "--episodes", "60"]
    )

    # Assert
    assert args.baseline == "rule"
    assert args.spacing == 6
    assert args.sources == [1, 3]
    assert args.goal == "estimate"
    assert args.goal_mean_std == 0.26
    assert args.known_std == 0.15
    assert args.episodes == 60


def test_parse_args_reads_the_unknown_area_check(script: ModuleType) -> None:
    # Act
    defaults = script.parse_args([])
    args = script.parse_args(["--unknown-std", "0.7", "--max-unknown-cells", "25"])

    # Assert
    assert (defaults.unknown_std, defaults.max_unknown_cells) == (0.8, 10)
    assert (args.unknown_std, args.max_unknown_cells) == (0.7, 25)


def test_test_agent_runs_a_whole_evaluation_without_model_files(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange: the real env and the rule baseline, with scripted options instead of the trained models
    base = HUGIN()
    base.train = True
    base.use_c_map = True
    base.GP_ON = True
    base.HRL = True
    base.agent_type = "META"  # type: ignore[attr-defined]  # set by the scripts, not in HUGIN.__init__
    base.spawn_at_survey_start = True
    base.vary_plume_size = True
    gp_env = GPWrapper(base, HRL=True)
    options = [SweepOption(base, gp_env, spacing=4), GasFillOption(base, gp_env), GasFillOption(base, gp_env)]
    env = HierarchicalEnv(gp_env, options, gamma=0.998, MAX_OPT_LENGTH=3, ACCURACY_GOALS=[0.9, 0.9, 0.9], GP=True, max_primitive_steps=480)
    survey = script.Survey(env, make_baseline("rule", env, seed=0), None, dict(script.UNTUNED_VARIANT), [1, 1], 160)
    monkeypatch.setattr(script, "make_survey", lambda *args: survey)

    # Act
    script.test_agent(Path("not_used"), episodes=2, baseline="rule", spacing=4)

    # Assert
    output = capsys.readouterr().out
    assert "2 episodes" in output
    assert "PLUME OVERLAP" in output
    assert "sweep_3" in output


def test_parse_args_reads_the_scripted_gas_option(script: ModuleType) -> None:
    # Act and Assert
    assert script.parse_args([]).scripted_gas is False
    assert script.parse_args(["--scripted-gas"]).scripted_gas is True


@pytest.mark.parametrize("argv", [["--baseline", "oracle"], ["--goal", "guess"], ["--sources", "1"]])
def test_parse_args_rejects_unknown_survey_settings(script: ModuleType, argv: list[str]) -> None:
    # Act and assert
    with pytest.raises(SystemExit):
        script.parse_args(argv)
