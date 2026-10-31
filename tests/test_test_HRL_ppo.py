"""Tests for the command line of HUGIN_gym/evaluation/test_HRL(ppo).py."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

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
