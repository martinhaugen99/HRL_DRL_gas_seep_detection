"""Summarise training runs as a Markdown table for docs/training_documentation.md.

Reads training_stats.pkl (written by EpisodeStatsCallback) and training_config.json from
each run folder and reports the mean ± std over the last training episodes. Runs that
share algorithm and tag (the folder name without the job ID) get an extra row with the
mean ± std of their run means.

Run from the repository root, e.g.:
    .venv/bin/python -m HUGIN_gym.evaluation.summarise_training_stats \
        ../trained_agents/*_compare_42235* --last 1000
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ENDED_EARLY = "Ended early (%)"


def load_run(run_dir: Path) -> tuple[dict, int]:
    """Return the episode stats and the maximum episode length of one run folder."""
    with (run_dir / "training_stats.pkl").open("rb") as f:
        stats = pickle.load(f)
    config = json.loads((run_dir / "training_config.json").read_text())
    return stats, config["ENVIRONMENT"]["max_episode_length"]


def episode_metrics(
    stats: dict, max_episode_length: int, last: int
) -> dict[str, np.ndarray]:
    """Per-episode metrics of the last `last` episodes, each stats list sliced on its own."""
    lengths = np.asarray(stats["episode_lengths"][-last:], dtype=float)
    return {
        "Total reward": np.array([sum(r) for r in stats["episode_rewards"][-last:]]),
        "Episode length": lengths,
        "Visited (%)": 100 * np.asarray(stats["visited_states_counts"][-last:], float),
        "Plume coverage (%)": 100
        * np.asarray(stats["episode_counter_above_threshold"][-last:], float),
        # the plume agent only ends early when it reaches its plume coverage goal
        ENDED_EARLY: 100 * (lengths < max_episode_length),
    }


def format_mean_std(values: np.ndarray, with_std: bool = True) -> str:
    """Format the mean (± std) of values, or '-' when there are none."""
    if values.size == 0:
        return "-"
    if not with_std:
        return f"{values.mean():.1f}"
    return f"{values.mean():.1f} ± {values.std():.1f}"


def group_name(run_name: str) -> str:
    """Strip the Slurm job ID: DQN_compare_4223550 -> DQN_compare."""
    return run_name.rsplit("_", 1)[0]


def to_markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    """Build a Markdown table from a header and rows of cells."""
    lines = [f"| {' | '.join(columns)} |", f"|{'---|' * len(columns)}"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a Markdown table summarising training runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="run folders, e.g. ../trained_agents/DQN_compare_4223550",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=1000,
        help="number of final training episodes to average over (default: 1000)",
    )
    args = parser.parse_args()

    # run name -> (number of episodes, metrics), grouped by algorithm and tag in input order
    groups: dict[str, dict[str, tuple[int, dict[str, np.ndarray]]]] = {}
    metric_names: list[str] = []
    for run_dir in args.run_dirs:
        stats, max_episode_length = load_run(run_dir)
        metrics = episode_metrics(stats, max_episode_length, args.last)
        metric_names = list(metrics)
        runs = groups.setdefault(group_name(run_dir.name), {})
        runs[run_dir.name] = (len(stats["episode_lengths"]), metrics)

    rows = []
    for group, runs in groups.items():
        for run_name, (n_episodes, metrics) in runs.items():
            cells = [
                format_mean_std(values, with_std=name != ENDED_EARLY)
                for name, values in metrics.items()
            ]
            rows.append([f"`{run_name}`", f"{n_episodes:,}", *cells])
        if len(runs) > 1:
            run_means = {
                name: np.array([m[name].mean() for _, m in runs.values()])
                for name in metric_names
            }
            cells = [f"**{format_mean_std(v)}**" for v in run_means.values()]
            rows.append([f"**{group} (mean of {len(runs)} runs)**", "", *cells])

    print(f"Mean ± std over the last {args.last:,} training episodes of each run.\n")
    print(to_markdown_table(["Run", "Episodes", *metric_names], rows))


if __name__ == "__main__":
    main()
