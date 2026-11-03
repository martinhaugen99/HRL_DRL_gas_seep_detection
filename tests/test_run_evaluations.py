"""Tests for HUGIN_gym/evaluation/run_evaluations.py: the matrix and running one evaluation, without model files."""

import subprocess
from pathlib import Path

import pytest

from HUGIN_gym.evaluation import run_evaluations as matrix


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        (
            matrix.Evaluation("sources_1_3", "sweep6_est", "sampled", 2, "pattern"),
            {
                "scenario": "sources_1_3",
                "survey": "sweep6_est",
                "actions": "sampled",
                "seed": 2,
            },
        ),
        (
            matrix.Evaluation(
                "small_plumes", "rule_sweep6_handover_estimate", "scripted"
            ),
            {
                "scenario": "small_plumes",
                "survey": "rule_sweep6_handover_estimate",
                "actions": "scripted",
                "seed": None,
            },
        ),
    ],
)
def test_parse_name_reads_what_the_name_holds(
    evaluation: matrix.Evaluation, expected: dict
) -> None:
    # Act
    parts = matrix.parse_name(evaluation.name)

    # Assert
    assert parts == expected


def test_build_evaluations_names_are_unique() -> None:
    # Act
    names = [evaluation.name for evaluation in matrix.build_evaluations()]

    # Assert
    assert len(names) == len(set(names))


def test_build_evaluations_has_every_variant_seed_and_action_in_the_main_comparison() -> (
    None
):
    # Act
    main = [
        e
        for e in matrix.build_evaluations()
        if e.scenario == "sources_1_3" and e.seed is not None
    ]

    # Assert
    assert len(main) == len(matrix.VARIANTS) * len(matrix.SEEDS) * 2
    assert {e.survey for e in main} == set(matrix.VARIANTS)


def test_build_evaluations_has_every_baseline_in_the_main_comparison() -> None:
    # Act
    main = {
        e.survey
        for e in matrix.build_evaluations()
        if e.scenario == "sources_1_3" and e.actions == "scripted"
    }

    # Assert
    assert main == set(matrix.BASELINES)


def test_every_command_states_its_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: without --sources, sweep6_est_one would be tested on the one source it trained on
    monkeypatch.setattr(
        matrix, "find_model", lambda agents_dir, pattern: Path("model.zip")
    )

    # Act
    commands = [
        matrix.command(e, tmp_path, tmp_path) for e in matrix.build_evaluations()
    ]

    # Assert
    assert all("--sources" in command for command in commands)


def test_the_one_source_variant_is_compared_on_one_to_three_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(
        matrix, "find_model", lambda agents_dir, pattern: Path("model.zip")
    )
    evaluation = matrix.Evaluation(
        "sources_1_3", "sweep6_est_one", "sampled", 1, "pattern"
    )

    # Act
    command = matrix.command(evaluation, tmp_path, tmp_path)

    # Assert
    assert command[command.index("--sources") + 1 : command.index("--sources") + 3] == [
        "1",
        "3",
    ]


@pytest.mark.parametrize(
    ("scenario", "low", "high"),
    [("small_plumes", "1", "2"), ("large_plumes", "5", "8")],
)
def test_plume_size_scenarios_set_the_width(
    scenario: str, low: str, high: str, tmp_path: Path
) -> None:
    # Act
    command = matrix.command(
        matrix.Evaluation(scenario, "lawnmower_sweep2_visit", "scripted"),
        tmp_path,
        tmp_path,
    )

    # Assert
    start = command.index("--sigma-range")
    assert command[start + 1 : start + 3] == [low, high]
    assert command[command.index("--episodes") + 1] == str(matrix.EPISODES_PLUME_SIZE)


@pytest.mark.parametrize(
    ("actions", "mode"), [("sampled", "4"), ("deterministic", "1"), ("scripted", "1")]
)
def test_command_uses_the_mode_of_the_actions(
    actions: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(
        matrix, "find_model", lambda agents_dir, pattern: Path("model.zip")
    )
    if actions == "scripted":
        evaluation = matrix.Evaluation("sources_1_3", "lawnmower_sweep2_visit", actions)
    else:
        evaluation = matrix.Evaluation(
            "sources_1_3", "sweep6_est", actions, 1, "pattern"
        )

    # Act
    command = matrix.command(evaluation, tmp_path, tmp_path)

    # Assert
    assert command[command.index("--mode") + 1] == mode


def test_command_episodes_override_is_for_a_quick_check(tmp_path: Path) -> None:
    # Act
    command = matrix.command(
        matrix.Evaluation("sources_1_3", "lawnmower_sweep2_visit", "scripted"),
        tmp_path,
        tmp_path,
        episodes=3,
    )

    # Assert
    assert command[command.index("--episodes") + 1] == "3"


def make_run(agents_dir: Path, name: str) -> Path:
    folder = agents_dir / name
    folder.mkdir(parents=True)
    model = folder / "HRL_PPO_scratch.zip"
    model.touch()
    return model


def test_find_model_finds_the_one_run_of_a_variant_and_seed(tmp_path: Path) -> None:
    # Arrange: sweep6_est must not match sweep6_est_map, and seed 1 must not match seed 2
    make_run(tmp_path, "HRL_PPO_sweep6_est_map_seed1_4387799")
    make_run(tmp_path, "HRL_PPO_sweep6_est_seed2_4387794")
    expected = make_run(tmp_path, "HRL_PPO_sweep6_est_seed1_4387793")

    # Act
    model = matrix.find_model(tmp_path, "HRL_PPO_sweep6_est_seed1_*")

    # Assert
    assert model == expected


def test_find_model_rejects_a_missing_run(tmp_path: Path) -> None:
    # Act and assert
    with pytest.raises(LookupError, match="found 0"):
        matrix.find_model(tmp_path, "HRL_PPO_sweep6_est_seed1_*")


def test_find_model_rejects_a_run_that_is_not_unique(tmp_path: Path) -> None:
    # Arrange
    make_run(tmp_path, "HRL_PPO_sweep6_est_seed1_4387793")
    make_run(tmp_path, "HRL_PPO_sweep6_est_seed1_4399999")

    # Act and assert
    with pytest.raises(LookupError, match="found 2"):
        matrix.find_model(tmp_path, "HRL_PPO_sweep6_est_seed1_*")


BASELINE = matrix.Evaluation("sources_1_3", "lawnmower_sweep2_visit", "scripted")


def test_run_one_skips_an_evaluation_with_a_saved_result(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / f"{BASELINE.name}.json").write_text("{}")

    def not_called(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise AssertionError("must not run")

    # Act
    result = matrix.run_one(BASELINE, tmp_path, tmp_path, run=not_called)

    # Assert
    assert result == (BASELINE.name, "skipped")


def test_run_one_reports_a_finished_evaluation(tmp_path: Path) -> None:
    # Arrange
    def finish(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        (tmp_path / f"{BASELINE.name}.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    # Act
    result = matrix.run_one(BASELINE, tmp_path, tmp_path, run=finish)

    # Assert
    assert result == (BASELINE.name, "done")


@pytest.mark.parametrize("writes_result", [True, False])
def test_run_one_reports_a_failure(tmp_path: Path, writes_result: bool) -> None:
    # Arrange: a crash, or an exit code of 0 without a result
    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if writes_result:
            (tmp_path / f"{BASELINE.name}.json").write_text("{}")
        return subprocess.CompletedProcess(command, 1 if writes_result else 0)

    # Act
    result = matrix.run_one(BASELINE, tmp_path, tmp_path, run=fail)

    # Assert
    assert result == (BASELINE.name, "FAILED")


def test_process_env_keeps_each_process_to_one_math_thread() -> None:
    # Act
    env = matrix.process_env()

    # Assert
    assert env["OMP_NUM_THREADS"] == "1"
    assert str(matrix.REPOSITORY) in env["PYTHONPATH"]


def test_command_passes_an_evaluation_seed_only_when_one_is_asked_for(
    tmp_path: Path,
) -> None:
    # Arrange
    evaluation = matrix.Evaluation(
        "sources_1_3", "lawnmower_sweep3_estimate", "scripted"
    )

    # Act
    default = matrix.command(evaluation, tmp_path, tmp_path)
    other_episodes = matrix.command(evaluation, tmp_path, tmp_path, seed=1)

    # Assert
    assert "--seed" not in default
    assert other_episodes[-2:] == ["--seed", "1"]
