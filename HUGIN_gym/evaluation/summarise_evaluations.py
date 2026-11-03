"""Tables from the results of run_evaluations.py: one CSV with every evaluation and one Markdown summary.

The summary follows the questions of run 6: which survey does best, does a variant beat the lawnmower and the
rule on every seed, how do the choices of the meta agent divide the survey, and did all evaluations of a
scenario get the same episodes.

Run from the repository root:
    .venv/bin/python -m HUGIN_gym.evaluation.summarise_evaluations ../eval-results/run6 --output docs/evaluation_results/run6
"""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from HUGIN_gym.evaluation.run_evaluations import parse_name

GOAL_ENDED = (
    0.99  # a survey must end by the goal in at least this share of the episodes
)
BOOTSTRAP_RESAMPLES = 2000
PHASES = ("early", "middle", "late")  # thirds of each survey

# The surveys a variant must beat, on the same goal
REFERENCES = {
    "sweep6_est": ("lawnmower_sweep3_estimate", "rule_sweep6_handover_estimate"),
    "sweep6_est_map": ("lawnmower_sweep3_estimate", "rule_sweep6_handover_estimate"),
    "sweep6_est_no_time": (
        "lawnmower_sweep3_estimate",
        "rule_sweep6_handover_estimate",
    ),
    "sweep6_est_one": ("lawnmower_sweep3_estimate", "rule_sweep6_handover_estimate"),
    "sweep6_est_control": (
        "lawnmower_sweep3_estimate",
        "rule_sweep6_handover_estimate",
    ),
    "space_est": ("lawnmower_sweep3_estimate", "rule_space_handover_estimate"),
    "sweep4_visit": ("lawnmower_sweep2_visit", "rule_sweep4_handover_visit"),
}

METRICS = (
    ("ended", "Ended by goal (%)", 1),
    ("steps", "AUV steps", 1),
    ("plume_overlap", "Overlap plume", 3),
    ("border_overlap", "Overlap border", 3),
    ("plume_visited", "Visited plume (%)", 1),
    ("border_visited", "Visited border (%)", 1),
    ("plumes_found", "Plumes found (%)", 1),
    ("turns", "Turns", 1),
)


@dataclass
class Result:
    """The saved evaluation and its name parts."""

    name: str
    scenario: str
    survey: str
    actions: str
    seed: int | None
    settings: dict
    action_names: list[str]
    episodes: list[dict]


def load_results(results_dir: Path) -> list[Result]:
    """Read every saved evaluation of a folder."""
    results = []
    for path in sorted(results_dir.glob("*.json")):
        saved = json.loads(path.read_text())
        parts = parse_name(path.stem)
        results.append(
            Result(
                path.stem,
                str(parts["scenario"]),
                str(parts["survey"]),
                str(parts["actions"]),
                parts["seed"] if isinstance(parts["seed"], int) else None,
                saved["settings"],
                saved["action_names"],
                saved["episodes"],
            )
        )
    return results


def values(episodes: list[dict], key: str) -> np.ndarray:
    return np.array([episode[key] for episode in episodes], dtype=float)


def switches_per_100_decisions(episodes: list[dict]) -> float:
    """How often the meta agent changes its choice: the oscillation that a supervisor can fall into."""
    changes = sum(
        sum(
            a != b
            for a, b in zip(episode["actions"], episode["actions"][1:], strict=False)
        )
        for episode in episodes
    )
    decisions = sum(len(episode["actions"]) for episode in episodes)
    return 100 * changes / max(decisions, 1)


def choice_shares(episodes: list[dict], n_actions: int) -> np.ndarray:
    """The share of all decisions per action."""
    counts = np.zeros(n_actions)
    for episode in episodes:
        counts += np.bincount(episode["actions"], minlength=n_actions)[:n_actions]
    return counts / max(counts.sum(), 1)


def choice_timeline(episodes: list[dict], n_actions: int, n_bins: int) -> np.ndarray:
    """The share per action in n_bins equal parts of each survey, shape (n_bins, n_actions).

    Surveys differ in length, so the position is the share of that survey's own decisions.
    """
    counts = np.zeros((n_bins, n_actions))
    for episode in episodes:
        n_decisions = len(episode["actions"])
        for index, action in enumerate(episode["actions"]):
            counts[index * n_bins // n_decisions, action] += 1
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)


def metrics(episodes: list[dict]) -> dict[str, float]:
    """The numbers of one evaluation, or of one group of its episodes."""
    n_plumes = values(episodes, "plumes").sum()
    return {
        "episodes": len(episodes),
        "ended": 100 * values(episodes, "ended_early").mean(),
        "steps": values(episodes, "steps").mean(),
        "plume_overlap": values(episodes, "plume_overlap").mean(),
        "border_overlap": values(episodes, "border_overlap").mean(),
        "plume_visited": 100 * values(episodes, "true_plume_visited").mean(),
        "border_visited": 100 * values(episodes, "true_border_visited").mean(),
        "plumes_found": 100 * values(episodes, "plumes_found").sum() / max(n_plumes, 1),
        "plumes_visited": 100
        * values(episodes, "plumes_visited").sum()
        / max(n_plumes, 1),
        "turns": values(episodes, "turns").mean(),
        "decisions": values(episodes, "decisions").mean(),
        "rmse": values(episodes, "rmse").mean(),
        "switches": switches_per_100_decisions(episodes),
    }


def paired_steps(
    episodes: list[dict], reference: list[dict], seed: int = 0
) -> tuple[float, float, float]:
    """Mean difference in AUV steps against a reference on the same episodes, and its 95 % bootstrap interval."""
    differences = values(episodes, "steps") - values(reference, "steps")
    rng = np.random.default_rng(seed)
    picks = rng.integers(len(differences), size=(BOOTSTRAP_RESAMPLES, len(differences)))
    means = differences[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(differences.mean()), float(low), float(high)


def beats(
    result: dict[str, float], reference: dict[str, float], steps_high: float
) -> bool:
    """The test written down before the results: fewer steps, ended by the goal and a map at least as good.

    Fewer steps means the whole interval of the paired difference is below zero. Maps are compared at the three
    decimals they are reported with, so a difference too small to report does not decide the outcome.
    """
    keys = ("plume_overlap", "border_overlap")
    return (
        steps_high < 0
        and result["ended"] >= 100 * GOAL_ENDED
        and all(round(result[key], 3) >= round(reference[key], 3) for key in keys)
        and round(result["plumes_found"], 1) >= round(reference["plumes_found"], 1)
    )


def same_episodes(results: list[Result]) -> dict[str, bool]:
    """Per scenario: did every evaluation get the same plumes in the same order?"""
    keys_per_scenario: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for result in results:
        keys = tuple(str(episode["plume_key"]) for episode in result.episodes)
        keys_per_scenario[result.scenario].add(keys)
    return {scenario: len(keys) == 1 for scenario, keys in keys_per_scenario.items()}


def format_group(group: list[float], digits: int) -> str:
    """One value, or the mean and std over the training seeds."""
    if len(group) == 1:
        return f"{group[0]:.{digits}f}"
    return f"{np.mean(group):.{digits}f} ± {np.std(group):.{digits}f}"


def table(columns: list[str], rows: list[list[str]]) -> str:
    lines = [f"| {' | '.join(columns)} |", f"|{'---|' * len(columns)}"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def group_results(results: list[Result]) -> dict[tuple[str, str], list[Result]]:
    """The evaluations of one survey and one kind of actions, over the training seeds."""
    groups: dict[tuple[str, str], list[Result]] = defaultdict(list)
    for result in results:
        groups[(result.survey, result.actions)].append(result)
    return groups


def survey_table(results: list[Result]) -> str:
    """Mean ± std over the training seeds for the trained variants, one value for the others."""
    rows = []
    for (survey, actions), group in group_results(results).items():
        numbers = [metrics(result.episodes) for result in group]
        cells = [
            format_group([n[key] for n in numbers], digits)
            for key, _, digits in METRICS
        ]
        rows.append([f"`{survey}`", actions, str(len(group)), *cells])
    return table(
        ["Survey", "Actions", "Seeds", *[label for _, label, _ in METRICS]], rows
    )


def steps_by_plumes_table(results: list[Result]) -> str:
    """Does the survey use more steps with more plumes, and does it still find them?"""
    rows = []
    for (survey, actions), group in group_results(results).items():
        cells = []
        for n_plumes in (1, 2, 3):
            parts = [
                [
                    episode
                    for episode in result.episodes
                    if episode["plumes"] == n_plumes
                ]
                for result in group
            ]
            parts = [part for part in parts if part]
            cells.append(
                format_group([metrics(part)["steps"] for part in parts], 0)
                if parts
                else "-"
            )
            cells.append(
                format_group([metrics(part)["plumes_found"] for part in parts], 1)
                if parts
                else "-"
            )
        rows.append([f"`{survey}`", actions, *cells])
    columns = ["Survey", "Actions"]
    for n_plumes in (1, 2, 3):
        columns += [f"Steps, {n_plumes} plume(s)", f"Found (%), {n_plumes} plume(s)"]
    return table(columns, rows)


def comparison_table(results: list[Result]) -> str:
    """Each seed of each variant against its references, with paired differences in AUV steps."""
    by_name = {
        (result.survey, result.actions, result.seed): result
        for result in results
        if result.seed is not None or result.actions == "scripted"
    }
    rows = []
    for (survey, actions, seed), result in by_name.items():
        if survey not in REFERENCES:
            continue
        numbers = metrics(result.episodes)
        for reference_name in REFERENCES[survey]:
            reference = by_name.get((reference_name, "scripted", None))
            if reference is None:
                continue
            mean, low, high = paired_steps(result.episodes, reference.episodes)
            verdict = beats(numbers, metrics(reference.episodes), high)
            rows.append(
                [
                    f"`{survey}`",
                    actions,
                    str(seed),
                    f"`{reference_name}`",
                    f"{mean:+.1f} ({low:+.1f} to {high:+.1f})",
                    f"{numbers['ended']:.1f}",
                    f"{numbers['plume_overlap']:.3f} / {numbers['border_overlap']:.3f}",
                    f"{numbers['plumes_found']:.1f}",
                    "yes" if verdict else "no",
                ]
            )
    return table(
        [
            "Variant",
            "Actions",
            "Seed",
            "Against",
            "Difference in AUV steps (95 % interval)",
            "Ended (%)",
            "Overlap plume / border",
            "Plumes found (%)",
            "Beats it",
        ],
        rows,
    )


def choices_table(results: list[Result]) -> str:
    """How the choices divide the survey between searching and mapping gas, overall and per third of the survey."""
    rows = []
    for (survey, actions), group in group_results(results).items():
        n_actions = len(group[0].action_names)
        if (
            n_actions != 3
        ):  # the variants with several option lengths are not part of run 6
            continue
        shares = np.mean(
            [choice_shares(result.episodes, n_actions) for result in group], axis=0
        )
        timeline = np.mean(
            [
                choice_timeline(result.episodes, n_actions, len(PHASES))
                for result in group
            ],
            axis=0,
        )
        switches = [metrics(result.episodes)["switches"] for result in group]
        rows.append(
            [
                f"`{survey}`",
                actions,
                *[f"{share:.2f}" for share in shares],
                *[f"{timeline[phase, 0]:.2f}" for phase in range(len(PHASES))],
                format_group(switches, 1),
            ]
        )
    return table(
        [
            "Survey",
            "Actions",
            "Search",
            "Plume",
            "Border",
            *[f"Search, {phase} third" for phase in PHASES],
            "Switches per 100 decisions",
        ],
        rows,
    )


def write_csv(results: list[Result], path: Path) -> None:
    """One row per evaluation, as the table view of the figures."""
    rows = []
    for result in results:
        n_actions = len(result.action_names)
        shares = choice_shares(result.episodes, n_actions)
        rows.append(
            {
                "name": result.name,
                "scenario": result.scenario,
                "survey": result.survey,
                "actions": result.actions,
                "seed": result.seed,
                **metrics(result.episodes),
                **{
                    f"share_{name}": share
                    for name, share in zip(result.action_names, shares, strict=True)
                },
            }
        )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summary(results: list[Result]) -> str:
    """The whole Markdown summary."""
    sections = ["# Results of run 6, from summarise_evaluations.py"]
    checks = same_episodes(results)
    sections.append(
        "**Same episodes per scenario:** "
        + ", ".join(
            f"{scenario} {'yes' if same else 'NO'}" for scenario, same in checks.items()
        )
    )
    for scenario in dict.fromkeys(result.scenario for result in results):
        group = [result for result in results if result.scenario == scenario]
        first = group[0].settings
        sections += [
            f"## {scenario}: sources {first['sources']}, plume width {first['sigma_range']}",
            "### Surveys",
            survey_table(group),
            "### Steps and plumes found per number of true plumes",
            steps_by_plumes_table(group),
            "### Each seed against the lawnmower and the rule",
            comparison_table(group),
            "### Choices of the meta agent",
            choices_table(group),
        ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tables from the results of run_evaluations.py."
    )
    parser.add_argument(
        "results_dir", type=Path, help="the folder with the .json results"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="folder for evaluations.csv and summary.md",
    )
    args = parser.parse_args()
    results = load_results(args.results_dir)
    if not results:
        raise SystemExit(f"No results in {args.results_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(results, args.output / "evaluations.csv")
    (args.output / "summary.md").write_text(summary(results), encoding="utf-8")
    print(f"{len(results)} evaluations summarised in {args.output}")


if __name__ == "__main__":
    main()
