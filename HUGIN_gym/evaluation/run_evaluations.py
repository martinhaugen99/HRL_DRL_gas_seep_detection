"""Evaluate the meta agents of run 6 and the baselines: every run of the matrix, each in its own process.

Each evaluation runs test_HRL(ppo).py and saves <name>.json (settings and every episode) and <name>.log in the
output folder. An evaluation with a saved result is skipped, so a stopped job can simply be submitted again.
Read the results with summarise_evaluations.py and plot_evaluations.py.

Run from the repository root, on Fox with fox_eval_HRL.sh, or on a PC:
    PYTHONPATH=. .venv/Scripts/python -m HUGIN_gym.evaluation.run_evaluations --workers 8
    ... --dry-run                      only list the evaluations
    ... --only "__rule_sweep6"         only the evaluations whose name matches this pattern
"""

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

SCRIPT = Path(__file__).with_name("test_HRL(ppo).py")
REPOSITORY = Path(__file__).parents[2]
SEPARATOR = "__"  # between the parts of an evaluation name

EPISODES = 1000
EPISODES_PLUME_SIZE = 500
SEEDS = (1, 2, 3)
VARIANTS = (
    "sweep6_est",
    "sweep4_visit",
    "sweep6_est_map",
    "sweep6_est_no_time",
    "space_est",
    "sweep6_est_one",
    "sweep6_est_control",
)

# Run 4 and run 5 models, the state of the project before the changes of run 6
EARLIER_RUNS = {
    "untuned": "HRL_PPO_untuned_4363368",
    "minimal_values": "HRL_PPO_minimal_values_4371280",
}

# Every evaluation states its sources: without it, a model is tested on the sources it
# trained on, so sweep6_est_one would get easier episodes than the other variants.
SOURCES_1_3 = ("--sources", "1", "3")
ONE_SOURCE = ("--sources", "1", "1")
SCENARIOS: dict[str, tuple[tuple[str, ...], int]] = {
    "sources_1_3": (SOURCES_1_3, EPISODES),
    "one_source": (ONE_SOURCE, EPISODES),
    "small_plumes": ((*SOURCES_1_3, "--sigma-range", "1", "2"), EPISODES_PLUME_SIZE),
    "large_plumes": ((*SOURCES_1_3, "--sigma-range", "5", "8"), EPISODES_PLUME_SIZE),
}

ESTIMATE = ("--goal", "estimate", "--goal-mean-std", "0.26", "--known-std", "0.1")
BASELINES: dict[str, tuple[str, ...]] = {
    # a lawnmower alone
    "lawnmower_sweep2_visit": ("--baseline", "lawnmower", "--spacing", "2"),
    "lawnmower_sweep3_estimate": (
        "--baseline",
        "lawnmower",
        "--spacing",
        "3",
        *ESTIMATE,
    ),
    # sweep 4 stops at the end of its path, with a survey about as long as the adaptive ones
    "lawnmower_sweep4_path_end": (
        "--baseline",
        "lawnmower",
        "--spacing",
        "4",
        "--goal",
        "estimate",
        "--goal-mean-std",
        "0.115",
        "--known-std",
        "0.5",
    ),
    # the sub-agents as the thesis uses them, without a sweep or a handover
    "rule_trained_visit": ("--baseline", "rule"),
    "random_trained_visit": ("--baseline", "random"),
    "space_only_visit": ("--baseline", "space"),
    # the rule with a sweep: without and with the handover, and with scripted gas handling
    "rule_sweep4_plume_border_visit": ("--baseline", "rule", "--spacing", "4"),
    "rule_sweep4_handover_visit": (
        "--baseline",
        "rule",
        "--spacing",
        "4",
        "--handover",
    ),
    "rule_sweep4_scripted_gas_visit": (
        "--baseline",
        "rule",
        "--spacing",
        "4",
        "--scripted-gas",
    ),
    "rule_sweep6_plume_border_estimate": (
        "--baseline",
        "rule",
        "--spacing",
        "6",
        *ESTIMATE,
    ),
    "rule_sweep6_handover_estimate": (
        "--baseline",
        "rule",
        "--spacing",
        "6",
        "--handover",
        *ESTIMATE,
    ),
    "rule_sweep6_scripted_gas_estimate": (
        "--baseline",
        "rule",
        "--spacing",
        "6",
        "--scripted-gas",
        *ESTIMATE,
    ),
    # the reference for space_est: the rule with the learned space agent
    "rule_space_handover_estimate": ("--baseline", "rule", "--handover", *ESTIMATE),
    # chance level inside the surveys of run 6
    "random_sweep4_handover_visit": (
        "--baseline",
        "random",
        "--spacing",
        "4",
        "--handover",
    ),
    "random_sweep6_handover_estimate": (
        "--baseline",
        "random",
        "--spacing",
        "6",
        "--handover",
        *ESTIMATE,
    ),
}
BASELINES_ONE_SOURCE = (
    "lawnmower_sweep2_visit",
    "lawnmower_sweep3_estimate",
    "rule_trained_visit",
    "random_trained_visit",
    "rule_sweep6_handover_estimate",
)
BASELINES_PLUME_SIZE = (
    "lawnmower_sweep2_visit",
    "lawnmower_sweep3_estimate",
    "rule_sweep6_handover_estimate",
    "rule_sweep6_scripted_gas_estimate",
)


@dataclass(frozen=True)
class Evaluation:
    """One run of test_HRL(ppo).py."""

    scenario: str
    survey: str  # a variant of run 6, an earlier run or a baseline
    actions: str  # sampled, deterministic, or scripted for a baseline
    seed: int | None = None  # the training seed of a variant of run 6
    run_pattern: str | None = None  # the run folder of a trained model, a glob pattern

    @property
    def name(self) -> str:
        parts = [self.scenario, self.survey, self.actions]
        if self.seed is not None:
            parts.append(f"seed{self.seed}")
        return SEPARATOR.join(parts)


def parse_name(name: str) -> dict[str, str | int | None]:
    """The scenario, survey, actions and seed of an evaluation name."""
    scenario, survey, actions, *seed = name.split(SEPARATOR)
    return {
        "scenario": scenario,
        "survey": survey,
        "actions": actions,
        "seed": int(seed[0].removeprefix("seed")) if seed else None,
    }


def trained(
    scenario: str, survey: str, seed: int | None, run_pattern: str
) -> list[Evaluation]:
    """A trained model with sampled and with deterministic actions."""
    return [
        Evaluation(scenario, survey, actions, seed, run_pattern)
        for actions in ("sampled", "deterministic")
    ]


def run_6(scenario: str, variants: tuple[str, ...]) -> list[Evaluation]:
    """The variants of run 6, every seed."""
    return [
        evaluation
        for variant in variants
        for seed in SEEDS
        for evaluation in trained(
            scenario, variant, seed, f"HRL_PPO_{variant}_seed{seed}_*"
        )
    ]


def earlier_runs(scenario: str) -> list[Evaluation]:
    """The models of runs 4 and 5."""
    return [
        evaluation
        for name, folder in EARLIER_RUNS.items()
        for evaluation in trained(scenario, name, None, folder)
    ]


def baselines(scenario: str, names: tuple[str, ...]) -> list[Evaluation]:
    return [Evaluation(scenario, name, "scripted") for name in names]


def build_evaluations() -> list[Evaluation]:
    """The whole matrix. The main comparison has every variant, the others what their question needs."""
    plume_size = ("sweep6_est", "sweep6_est_map")
    evaluations = [
        *run_6("sources_1_3", VARIANTS),
        *earlier_runs("sources_1_3"),
        *baselines("sources_1_3", tuple(BASELINES)),
        # does training on several plumes matter, and the comparison with the thesis
        *run_6("one_source", ("sweep6_est", "sweep6_est_one")),
        *earlier_runs("one_source"),
        *baselines("one_source", BASELINES_ONE_SOURCE),
    ]
    for scenario in ("small_plumes", "large_plumes"):
        evaluations += [
            *run_6(scenario, plume_size),
            *trained(scenario, "minimal_values", None, EARLIER_RUNS["minimal_values"]),
            *baselines(scenario, BASELINES_PLUME_SIZE),
        ]
    return evaluations


def find_model(agents_dir: Path, run_pattern: str) -> Path:
    """The final model of the one run folder that matches. A run that is missing or not unique is an error."""
    models = sorted(agents_dir.glob(f"{run_pattern}/HRL_PPO_scratch.zip"))
    if len(models) != 1:
        raise LookupError(
            f"Expected one final model for {run_pattern} in {agents_dir}, found {len(models)}"
        )
    return models[0]


def command(
    evaluation: Evaluation,
    agents_dir: Path,
    output_dir: Path,
    episodes: int | None = None,
) -> list[str]:
    """The command line of one evaluation. episodes overrides the number of the scenario, for a quick check."""
    scenario_args, scenario_episodes = SCENARIOS[evaluation.scenario]
    mode = "4" if evaluation.actions == "sampled" else "1"
    if evaluation.run_pattern is None:
        survey_args = list(BASELINES[evaluation.survey])
    else:
        survey_args = ["--model", str(find_model(agents_dir, evaluation.run_pattern))]
    return [
        sys.executable,
        str(SCRIPT),
        "--mode",
        mode,
        "--episodes",
        str(episodes or scenario_episodes),
        "--results-json",
        str(output_dir / f"{evaluation.name}.json"),
        *scenario_args,
        *survey_args,
    ]


def process_env() -> dict[str, str]:
    """One math thread per process, because the evaluations already fill the cores."""
    python_path = os.pathsep.join(
        filter(None, [str(REPOSITORY), os.environ.get("PYTHONPATH")])
    )
    return {
        **os.environ,
        "PYTHONPATH": python_path,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "TQDM_DISABLE": "1",
    }


def run_one(
    evaluation: Evaluation,
    agents_dir: Path,
    output_dir: Path,
    episodes: int | None = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[str, str]:
    """Run one evaluation. Returns its name and "skipped", "done" or "FAILED"."""
    result = output_dir / f"{evaluation.name}.json"
    if result.exists():
        return evaluation.name, "skipped"
    with (output_dir / f"{evaluation.name}.log").open("w") as log:
        finished = run(
            command(evaluation, agents_dir, output_dir, episodes),
            cwd=REPOSITORY,
            env=process_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return (
        evaluation.name,
        "done" if finished.returncode == 0 and result.exists() else "FAILED",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the meta agents of run 6 and the baselines."
    )
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("../trained-agents"),
        help="the run folders (default: ../trained-agents)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../eval-results/run6"),
        help="where to save the results (default: ../eval-results/run6)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="evaluations at the same time (default: 4)",
    )
    parser.add_argument(
        "--only", help="only the evaluations whose name matches this regular expression"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="episodes per evaluation instead of the matrix's, for a quick check",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="only list the evaluations"
    )
    args = parser.parse_args()
    agents_dir, output_dir = args.agents_dir.resolve(), args.output_dir.resolve()

    evaluations = [
        evaluation
        for evaluation in build_evaluations()
        if args.only is None or re.search(args.only, evaluation.name)
    ]
    if args.dry_run:
        print("\n".join(evaluation.name for evaluation in evaluations))
        print(f"{len(evaluations)} evaluations")
        return

    # stop before any work if a model is missing
    missing = []
    for pattern in sorted({e.run_pattern for e in evaluations if e.run_pattern}):
        try:
            find_model(agents_dir, pattern)
        except LookupError as error:
            missing.append(str(error))
    if missing:
        sys.exit("\n".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = pool.map(
            lambda e: run_one(e, agents_dir, output_dir, args.episodes), evaluations
        )
        statuses = {}
        for count, (name, status) in enumerate(jobs, start=1):
            statuses[name] = status
            print(f"[{count}/{len(evaluations)}] {status}: {name}", flush=True)

    failed = [name for name, status in statuses.items() if status == "FAILED"]
    print(
        f"{len(statuses) - len(failed)} of {len(statuses)} evaluations have results in {output_dir}"
    )
    if failed:
        sys.exit(f"Failed, see the .log files: {', '.join(failed)}")


if __name__ == "__main__":
    main()
