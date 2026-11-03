"""Figures for the paper from the results of run_evaluations.py and the TensorBoard logs of run 6.

Figures for print, as PDF with embedded fonts and as PNG, sized for a column of the IEEE template:
    learning_curves   AUV steps and share ended by the goal against the decisions of training
    surveys           AUV steps of each survey, with every training seed as its own point
    choices           the share of each option over the survey, for the rule and a trained meta agent

Run from the repository root:
    .venv/bin/python -m HUGIN_gym.evaluation.plot_evaluations ../eval-results/run6 --agents-dir ../trained-agents --output docs/evaluation_results/run6
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from HUGIN_gym.evaluation.summarise_evaluations import (
    Result,
    choice_timeline,
    load_results,
    metrics,
)

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,  # embedded fonts, as the IEEE template asks for
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "text.color": "#0b0b0b",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLUMN_WIDTH = 3.5  # inches, one column of the IEEE template
GRID_COLOR = "#e1e0d9"
REFERENCE_COLOR = "#52514e"
# The first three slots of the reference palette keep their colors in every figure. Only
# these three are used, because they stay apart for readers with color vision deficiency.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")

LEARNING_VARIANTS = {
    "sweep6_est": "Repaired reward, patch bonus, gas state",
    "sweep6_est_map": "Reward for map gain",
    "sweep6_est_control": "Control: reward, bonus and state of run 5",
}
# Label, position along the axis, alignment and side of the line. The curves stay above the rule at the
# start and below the lawnmower at the end, so the labels sit there.
LEARNING_REFERENCES = {
    "rule_sweep6_handover_estimate": ("Rule, sweep 6", 0.02, "left", "top"),
    "lawnmower_sweep3_estimate": ("Lawnmower, sweep 3", 0.98, "right", "bottom"),
}
# The plateaus decide the comparison, so the panel leaves out the start of training, at 480 steps.
STEPS_AXIS_LIMITS = (205.0, 340.0)
SURVEY_ROWS = {
    "lawnmower_sweep3_estimate": "Lawnmower, sweep 3",
    "sweep6_est_control": "Control, sweep 6",
    "sweep6_est_map": "Meta agent, map gain",
    "sweep6_est_no_time": "Meta agent, no time input",
    "sweep6_est_one": "Meta agent, one-source training",
    "sweep6_est": "Meta agent, repaired reward",
    "space_est": "Meta agent, learned space agent",
    "rule_sweep6_handover_estimate": "Rule, sweep 6, handover",
    "rule_sweep6_scripted_gas_estimate": "Rule, sweep 6, scripted gas",
}
# Sampled and deterministic points of one survey can be equal, so they get their own line in the row.
ROW_OFFSETS = {"sampled": 0.17, "deterministic": -0.17, "scripted": 0.0}
OPTION_LABELS = ("Search", "Plume", "Border")
TIMELINE_BINS = 10
SMOOTHING_WINDOW = 25  # PPO updates, for the learning curves


def style_axes(ax: Axes) -> None:
    ax.grid(color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def smooth(series: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """A running mean with the same length, using a shorter window at the start."""
    totals = np.cumsum(np.insert(series, 0, 0.0))
    starts = np.maximum(np.arange(1, len(series) + 1) - window, 0)
    stops = np.arange(1, len(series) + 1)
    return (totals[stops] - totals[starts]) / (stops - starts)


def read_curve(run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """The decisions and values of one scalar in a run's TensorBoard folder."""
    events = sorted((run_dir / "tensorboard").glob("**/events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"No TensorBoard events in {run_dir}")
    steps, scalars = [], []
    for event_file in events:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag in accumulator.Tags()["scalars"]:
            for event in accumulator.Scalars(tag):
                steps.append(event.step)
                scalars.append(event.value)
    if not steps:
        raise ValueError(f"No {tag} in the TensorBoard events of {run_dir}")
    return np.asarray(steps, dtype=float), np.asarray(scalars, dtype=float)


def seed_band(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The mean, lowest and highest seed of smoothed curves, on the first curve's decisions."""
    grid = curves[0][0]
    stacked = np.array(
        [np.interp(grid, steps, smooth(values)) for steps, values in curves]
    )
    return grid, stacked.mean(axis=0), stacked.min(axis=0), stacked.max(axis=0)


def plot_learning_curves(agents_dir: Path, results: list[Result]) -> Figure:
    """AUV steps and ended share during training, three seeds per variant, with the rule and the lawnmower."""
    fig, (steps_ax, ended_ax) = plt.subplots(
        2,
        1,
        figsize=(COLUMN_WIDTH, 3.0),
        sharex=True,
        height_ratios=(2, 1),
        constrained_layout=True,
    )
    for (variant, label), color in zip(
        LEARNING_VARIANTS.items(), SERIES_COLORS, strict=True
    ):
        runs = sorted(agents_dir.glob(f"HRL_PPO_{variant}_seed?_*"))
        if not runs:
            raise FileNotFoundError(f"No runs of {variant} in {agents_dir}")
        for ax, tag, scale in (
            (steps_ax, "hrl/episode_steps", 1.0),
            (ended_ax, "hrl/ended_early", 100.0),
        ):
            grid, mean, low, high = seed_band([read_curve(run, tag) for run in runs])
            ax.fill_between(
                grid / 1e6,
                scale * low,
                scale * high,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(grid / 1e6, scale * mean, color=color, linewidth=1.2, label=label)
    for result in results:
        if result.scenario == "sources_1_3" and result.survey in LEARNING_REFERENCES:
            numbers = metrics(result.episodes)
            label, position, align, side = LEARNING_REFERENCES[result.survey]
            steps_ax.axhline(numbers["steps"], color=REFERENCE_COLOR, linewidth=0.6)
            steps_ax.annotate(
                label,
                xy=(position, numbers["steps"]),
                xycoords=("axes fraction", "data"),
                xytext=(0, 1.5 if side == "bottom" else -1.5),
                textcoords="offset points",
                ha=align,
                va=side,
                color=REFERENCE_COLOR,
                fontsize=6,
            )
    steps_ax.set_ylim(*STEPS_AXIS_LIMITS)
    steps_ax.set_ylabel("AUV steps per survey")
    ended_ax.set_ylabel("Ended (%)")
    ended_ax.set_xlabel("Decisions of training (millions)")
    handles, labels = steps_ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside upper center")
    for ax in (steps_ax, ended_ax):
        style_axes(ax)
    return fig


def plot_surveys(results: list[Result], scenario: str = "sources_1_3") -> Figure:
    """AUV steps per survey on the same episodes: every training seed is a point, sampled and deterministic apart.

    The rows run from the fewest steps at the top, by the best evaluation of each survey.
    """
    best_steps = {
        survey: min(
            metrics(r.episodes)["steps"]
            for r in results
            if r.survey == survey and r.scenario == scenario
        )
        for survey in SURVEY_ROWS
        if any(r.survey == survey and r.scenario == scenario for r in results)
    }
    rows = [
        (survey, SURVEY_ROWS[survey])
        for survey in sorted(best_steps, key=lambda survey: best_steps[survey])
    ]
    fig, ax = plt.subplots(
        figsize=(COLUMN_WIDTH, 0.3 + 0.2 * len(rows)), constrained_layout=True
    )
    positions = {survey: index for index, (survey, _) in enumerate(reversed(rows))}
    kinds = {
        "sampled": ("Sampled", SERIES_COLORS[0]),
        "deterministic": ("Deterministic", SERIES_COLORS[1]),
        "scripted": ("No learning", SERIES_COLORS[2]),
    }
    for actions, (label, color) in kinds.items():
        points = [
            (metrics(r.episodes)["steps"], positions[r.survey] + ROW_OFFSETS[actions])
            for r in results
            if r.scenario == scenario and r.actions == actions and r.survey in positions
        ]
        if points:
            x, y = zip(*points, strict=True)
            ax.scatter(
                x,
                y,
                s=12,
                color=color,
                edgecolors="white",
                linewidths=0.4,
                label=label,
                zorder=3,
            )
    ax.set_yticks(
        list(positions.values()), [dict(rows)[survey] for survey in positions]
    )
    ax.set_xlabel("AUV steps per survey, mean of 1,000 episodes")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncols=3)
    style_axes(ax)
    return fig


def plot_choices(
    results: list[Result],
    trained: str = "sweep6_est",
    reference: str = "rule_sweep6_handover_estimate",
    scenario: str = "sources_1_3",
) -> Figure:
    """The share of each option over the survey, per number of true plumes: the trained agent against the rule."""
    rows = {reference: "Rule", trained: "Trained, sampled"}
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(COLUMN_WIDTH, 2.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    progress = (np.arange(TIMELINE_BINS) + 0.5) / TIMELINE_BINS * 100
    for row_axes, (survey, row_label) in zip(axes, rows.items(), strict=True):
        group = [
            r
            for r in results
            if r.scenario == scenario
            and r.survey == survey
            and r.actions in ("sampled", "scripted")
        ]
        for ax, n_plumes in zip(row_axes, (1, 2, 3), strict=True):
            timelines = []
            for result in group:
                episodes = [e for e in result.episodes if e["plumes"] == n_plumes]
                if episodes:
                    timelines.append(
                        choice_timeline(
                            episodes, len(result.action_names), TIMELINE_BINS
                        )
                    )
            style_axes(ax)
            ax.set_ylim(0, 100)
            ax.set_xlim(progress[0], progress[-1])
            if timelines:
                shares = 100 * np.mean(timelines, axis=0)
                ax.stackplot(
                    progress,
                    shares.T,
                    colors=SERIES_COLORS,
                    labels=OPTION_LABELS,
                    edgecolor="white",
                    linewidth=0.5,
                )
            if ax in axes[0]:
                ax.set_title(f"{n_plumes} plume{'s' if n_plumes > 1 else ''}")
        row_axes[0].set_ylabel(row_label)
    fig.supylabel("Share of decisions (%)", fontsize=7, color=REFERENCE_COLOR)
    fig.supxlabel("Progress through the survey (%)", fontsize=7, color=REFERENCE_COLOR)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside upper center", ncols=3)
    return fig


def save(fig: Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figures for the paper from the results of run_evaluations.py."
    )
    parser.add_argument(
        "results_dir", type=Path, help="the folder with the .json results"
    )
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("../trained-agents"),
        help="the run folders with the TensorBoard logs",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="folder for the figures"
    )
    args = parser.parse_args()
    results = load_results(args.results_dir)
    save(plot_learning_curves(args.agents_dir, results), args.output, "learning_curves")
    save(plot_surveys(results), args.output, "surveys")
    save(plot_choices(results), args.output, "choices")
    print(f"Figures saved in {args.output}")


if __name__ == "__main__":
    main()
