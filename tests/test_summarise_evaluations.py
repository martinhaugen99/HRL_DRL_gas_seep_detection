"""Tests for HUGIN_gym/evaluation/summarise_evaluations.py on small results written by hand."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from HUGIN_gym.evaluation import summarise_evaluations as summarise


def episode(
    steps: float = 100,
    ended: float = 1.0,
    plumes: int = 1,
    found: int = 1,
    actions: list[int] | None = None,
    key: str = "a",
) -> dict:
    return {
        "steps": steps,
        "ended_early": ended,
        "plume_overlap": 0.99,
        "border_overlap": 0.98,
        "true_plume_visited": 0.9,
        "true_border_visited": 0.8,
        "plumes": plumes,
        "plumes_found": found,
        "plumes_visited": found,
        "turns": 10,
        "decisions": 3,
        "reward": -5.0,
        "rmse": 0.01,
        "plume_key": key,
        "actions": actions if actions is not None else [0, 1, 2],
    }


def save(folder: Path, name: str, episodes: list[dict]) -> None:
    settings = {"sources": [1, 3], "sigma_range": [2.0, 5.0]}
    (folder / f"{name}.json").write_text(
        json.dumps(
            {
                "settings": settings,
                "action_names": ["sweep_3", "plume_3", "border_3"],
                "episodes": episodes,
            }
        )
    )


def test_load_results_reads_the_name_parts_and_the_episodes(tmp_path: Path) -> None:
    # Arrange
    save(tmp_path, "sources_1_3__sweep6_est__sampled__seed2", [episode()])
    save(tmp_path, "sources_1_3__lawnmower_sweep3_estimate__scripted", [episode()])

    # Act
    results = {result.survey: result for result in summarise.load_results(tmp_path)}

    # Assert
    assert results["sweep6_est"].seed == 2
    assert results["sweep6_est"].actions == "sampled"
    assert results["lawnmower_sweep3_estimate"].seed is None
    assert len(results["sweep6_est"].episodes) == 1


def test_metrics_share_of_plumes_found_counts_plumes_not_episodes() -> None:
    # Arrange: 3 of 4 plumes found, in one episode of each kind
    episodes = [episode(plumes=3, found=2), episode(plumes=1, found=1)]

    # Act
    numbers = summarise.metrics(episodes)

    # Assert
    assert numbers["plumes_found"] == pytest.approx(75.0)


def test_metrics_converts_shares_to_percent() -> None:
    # Act
    numbers = summarise.metrics([episode(ended=1.0), episode(ended=0.0)])

    # Assert
    assert numbers["ended"] == pytest.approx(50.0)
    assert numbers["plume_visited"] == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("actions", "expected"),
    [([0, 0, 0, 0], 0.0), ([0, 1, 0, 1], 75.0), ([0, 0, 1, 1], 25.0)],
)
def test_switches_per_100_decisions_counts_changes_of_choice(
    actions: list[int], expected: float
) -> None:
    # Act and assert
    assert summarise.switches_per_100_decisions(
        [episode(actions=actions)]
    ) == pytest.approx(expected)


def test_choice_shares_sum_to_one_per_option() -> None:
    # Act
    shares = summarise.choice_shares([episode(actions=[0, 0, 1, 2])], 3)

    # Assert
    assert shares == pytest.approx([0.5, 0.25, 0.25])


def test_choice_timeline_uses_the_progress_of_each_survey() -> None:
    # Arrange: both surveys search in their first half, whatever their length
    episodes = [episode(actions=[0, 1]), episode(actions=[0, 0, 1, 1])]

    # Act
    timeline = summarise.choice_timeline(episodes, 3, 2)

    # Assert
    assert timeline[0] == pytest.approx([1.0, 0.0, 0.0])
    assert timeline[1] == pytest.approx([0.0, 1.0, 0.0])


def test_paired_steps_gives_the_mean_difference_and_a_narrow_interval_when_it_is_constant() -> (
    None
):
    # Arrange
    survey = [episode(steps=90), episode(steps=190)]
    reference = [episode(steps=100), episode(steps=200)]

    # Act
    mean, low, high = summarise.paired_steps(survey, reference)

    # Assert
    assert (mean, low, high) == pytest.approx((-10.0, -10.0, -10.0))


def test_paired_steps_interval_contains_zero_when_there_is_no_clear_difference() -> (
    None
):
    # Arrange
    survey = [episode(steps=90), episode(steps=110)] * 50
    reference = [episode(steps=100)] * 100

    # Act
    mean, low, high = summarise.paired_steps(survey, reference)

    # Assert
    assert mean == pytest.approx(0.0)
    assert low < 0 < high


GOOD = {
    "ended": 99.5,
    "plume_overlap": 0.998,
    "border_overlap": 0.991,
    "plumes_found": 100.0,
}


@pytest.mark.parametrize(
    ("changes", "steps_high", "expected"),
    [
        ({}, -1.0, True),
        ({}, 1.0, False),  # the interval of the difference in steps contains zero
        ({"ended": 98.9}, -1.0, False),
        ({"plume_overlap": 0.9960}, -1.0, False),
        (
            {"border_overlap": 0.9896},
            -1.0,
            True,
        ),  # the same value at three decimals as 0.990
        ({"plumes_found": 99.8}, -1.0, False),
    ],
)
def test_beats_needs_fewer_steps_ended_surveys_and_a_map_at_least_as_good(
    changes: dict, steps_high: float, expected: bool
) -> None:
    # Arrange
    reference = {
        "ended": 100.0,
        "plume_overlap": 0.997,
        "border_overlap": 0.990,
        "plumes_found": 100.0,
    }

    # Act and assert
    assert summarise.beats({**GOOD, **changes}, reference, steps_high) is expected


def result(
    survey: str,
    actions: str,
    seed: int | None,
    episodes: list[dict],
    scenario: str = "sources_1_3",
) -> summarise.Result:
    name = f"{scenario}__{survey}__{actions}" + (f"__seed{seed}" if seed else "")
    return summarise.Result(
        name,
        scenario,
        survey,
        actions,
        seed,
        {"sources": [1, 3], "sigma_range": [2.0, 5.0]},
        ["sweep_3", "plume_3", "border_3"],
        episodes,
    )


def test_same_episodes_sees_a_scenario_whose_plumes_differ() -> None:
    # Arrange
    results = [
        result("a", "scripted", None, [episode(key="x"), episode(key="y")]),
        result("b", "scripted", None, [episode(key="x"), episode(key="y")]),
        result("a", "scripted", None, [episode(key="x")], scenario="one_source"),
        result("b", "scripted", None, [episode(key="z")], scenario="one_source"),
    ]

    # Act
    checks = summarise.same_episodes(results)

    # Assert
    assert checks == {"sources_1_3": True, "one_source": False}


def test_format_group_gives_one_value_or_the_mean_and_std_over_the_seeds() -> None:
    # Act and assert
    assert summarise.format_group([12.34], 1) == "12.3"
    assert summarise.format_group([10.0, 20.0], 1) == "15.0 ± 5.0"


def test_survey_table_has_one_row_per_survey_and_actions_over_the_seeds() -> None:
    # Arrange
    results = [
        result("sweep6_est", "sampled", 1, [episode(steps=100)]),
        result("sweep6_est", "sampled", 2, [episode(steps=120)]),
        result("lawnmower_sweep3_estimate", "scripted", None, [episode(steps=280)]),
    ]

    # Act
    text = summarise.survey_table(results)

    # Assert
    assert "| `sweep6_est` | sampled | 2 |" in text
    assert "110.0 ± 10.0" in text
    assert "| `lawnmower_sweep3_estimate` | scripted | 1 |" in text


def test_comparison_table_compares_each_seed_with_each_reference() -> None:
    # Arrange
    episodes = [episode(steps=100) for _ in range(5)]
    results = [
        result("sweep6_est", "sampled", 1, episodes),
        result("sweep6_est", "sampled", 2, episodes),
        result(
            "lawnmower_sweep3_estimate",
            "scripted",
            None,
            [episode(steps=280) for _ in range(5)],
        ),
        result(
            "rule_sweep6_handover_estimate",
            "scripted",
            None,
            [episode(steps=100) for _ in range(5)],
        ),
    ]

    # Act
    rows = summarise.comparison_table(results).splitlines()[2:]

    # Assert
    assert len(rows) == 4
    assert "-180.0" in rows[0] and rows[0].endswith("yes |")
    assert "+0.0" in rows[1] and rows[1].endswith(
        "no |"
    )  # the same steps as the rule do not beat it


def test_steps_by_plumes_table_splits_the_episodes_by_true_plumes() -> None:
    # Arrange
    results = [
        result(
            "rule",
            "scripted",
            None,
            [episode(steps=100, plumes=1), episode(steps=200, plumes=2, found=2)],
        )
    ]

    # Act
    text = summarise.steps_by_plumes_table(results)

    # Assert
    assert "| `rule` | scripted | 100 | 100.0 | 200 | 100.0 | - | - |" in text


def test_write_csv_has_one_row_per_evaluation_with_the_choice_shares(
    tmp_path: Path,
) -> None:
    # Arrange
    path = tmp_path / "evaluations.csv"

    # Act
    summarise.write_csv(
        [result("rule", "scripted", None, [episode(actions=[0, 0, 1, 2])])], path
    )

    # Assert
    with path.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert float(rows[0]["share_sweep_3"]) == pytest.approx(0.5)


def test_summary_has_a_section_per_scenario_and_the_check_of_the_episodes() -> None:
    # Arrange
    results = [
        result("rule", "scripted", None, [episode()]),
        result("rule", "scripted", None, [episode()], scenario="one_source"),
    ]

    # Act
    text = summarise.summary(results)

    # Assert
    assert "## sources_1_3" in text
    assert "## one_source" in text
    assert "sources_1_3 yes" in text


def test_summary_starts_with_the_main_comparison() -> None:
    # Arrange: the files of the other scenarios sort first
    results = [
        result("rule", "scripted", None, [episode()], scenario="large_plumes"),
        result("rule", "scripted", None, [episode()]),
    ]

    # Act
    text = summarise.summary(results)

    # Assert
    assert text.index("## sources_1_3") < text.index("## large_plumes")


def test_mean_over_seeds_averages_the_steps_of_each_episode() -> None:
    # Arrange
    group = [
        result("sweep6_est", "sampled", seed, [episode(steps=steps, plumes=2, found=2)])
        for seed, steps in ((1, 100), (2, 110))
    ]

    # Act
    rows = summarise.mean_over_seeds(group)

    # Assert
    assert rows == [{"steps": 105.0, "plumes": 2}]


def test_contrast_table_gives_the_difference_in_total_and_per_number_of_plumes() -> (
    None
):
    # Arrange
    ours = [episode(steps=90), episode(steps=180, plumes=2, found=2)]
    rule = [episode(steps=100), episode(steps=200, plumes=2, found=2)]
    results = [
        result("sweep6_est", "sampled", 1, ours),
        result("rule_sweep6_handover_estimate", "scripted", None, rule),
    ]

    # Act
    text = summarise.contrast_table(results)

    # Assert
    assert "`sweep6_est`, sampled | `rule_sweep6_handover_estimate`, scripted" in text
    assert "-10.0 (-10.0 to -10.0)" in text  # one plume
    assert "-20.0 (-20.0 to -20.0)" in text  # two plumes
    assert text.rstrip().endswith("| - |")  # no episode with three plumes


def test_contrast_table_leaves_out_surveys_without_results() -> None:
    # Act
    text = summarise.contrast_table([result("rule", "scripted", None, [episode()])])

    # Assert
    assert "sweep6_est" not in text


def test_values_is_a_float_array() -> None:
    # Act
    steps = summarise.values([episode(steps=3)], "steps")

    # Assert
    assert steps.dtype == np.float64
