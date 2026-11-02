"""Compare the true map with the map the agent estimated, after an evaluation.

The agent never sees the true plume. It measures concentrations along its path, and the GP estimates the rest of the map.
The end goal is checked against that estimate, so the agent does not need to visit the whole map. The Meshcat view shows
the path and the true gas, but not the estimate. This script shows both, side by side.

It runs the episodes, then saves in the output folder:
- map_comparison.png and .svg: the worst, the 10th percentile, the median and the best episode
- map_comparison.csv: the results of every episode

Run from the repository root, e.g.:
    PYTHONPATH=. .venv/bin/python -m HUGIN_gym.evaluation.compare_maps \
        --model ../trained-agents/HRL_PPO_minimal_values_4371280/HRL_PPO_scratch.zip --episodes 100 --output ../map_comparison
"""

import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from stable_baselines3.common.utils import set_random_seed

from HUGIN_gym.evaluation.survey_metrics import gp_maps, map_metrics


def _load_evaluation_script() -> ModuleType:
    """The evaluation script as a module; its file name is not a valid module name."""
    spec = importlib.util.spec_from_file_location("test_HRL_ppo", Path(__file__).with_name("test_HRL(ppo).py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = _load_evaluation_script()

# Chart colors. Sub-agents and error classes are validated for color vision deficiency; the neutral gray is not a series.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
HAIRLINE = "#e1e0d9"
NEUTRAL = "#c3c2b7"
OPTION_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # first option (space or sweep), plume, border
MISSED_COLOR = "#e34948"
EXTRA_COLOR = "#6250d6"
CONCENTRATION_MAP = LinearSegmentedColormap.from_list(
    "concentration", [SURFACE, "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
)
UNCERTAINTY_MAP = LinearSegmentedColormap.from_list(
    "uncertainty", [SURFACE, "#fbd9ca", "#f2a27f", "#eb6834", "#b9481c", "#7c2c0c"]
)
COMPARISON_MAP = ListedColormap([SURFACE, NEUTRAL, MISSED_COLOR, EXTRA_COLOR])

NOT_GAS, CORRECT, MISSED, EXTRA = 0, 1, 2, 3
PICKED_QUANTILES = ((0.0, "Worst"), (0.1, "10th percentile"), (0.5, "Median"), (1.0, "Best"))
PANEL_TITLES = (
    "Path and visited cells",
    "True concentration",
    "Estimated concentration",
    "Plume: true against estimate",
    "Border: true against estimate",
    "GP std (how unsure it is)",
)


@dataclass
class EpisodeMaps:
    """The true map, the estimated map and the path at the end of an episode. All maps have shape (X, Y)."""

    true_concentration: np.ndarray
    gp_mean: np.ndarray
    gp_std: np.ndarray
    true_plume: np.ndarray
    true_border: np.ndarray
    visited: np.ndarray
    path: np.ndarray  # (steps + 1, 2): the x and y of the start and of every measurement
    path_options: np.ndarray  # (steps,): the option that made each move
    c_threshold: float
    c_width: float
    results: dict[str, float]

    @property
    def score(self) -> float:
        """How well the estimate matches the truth, to rank the episodes."""
        return (self.results["plume_overlap"] + self.results["border_overlap"]) / 2


def comparison_classes(true_map: np.ndarray, estimated_map: np.ndarray) -> np.ndarray:
    """NOT_GAS, CORRECT (in both maps), MISSED (only in the true map) or EXTRA (only in the estimate) per cell."""
    classes = np.full(true_map.shape, NOT_GAS, dtype=np.int8)
    classes[true_map & estimated_map] = CORRECT
    classes[true_map & ~estimated_map] = MISSED
    classes[~true_map & estimated_map] = EXTRA
    return classes


def pick_episodes(scores: list[float]) -> list[tuple[str, int]]:
    """A label and an index for each of PICKED_QUANTILES, without showing an episode twice."""
    order = np.argsort(scores, kind="stable")
    picked: dict[int, str] = {}
    for quantile, label in PICKED_QUANTILES:
        index = int(order[round(quantile * (len(order) - 1))])
        picked.setdefault(index, label)
    return [(label, index) for index, label in picked.items()]


def run_episode(env, model, deterministic: bool) -> EpisodeMaps:
    """Run one episode and keep what the plot needs."""
    gp_env = env.base_env
    base = gp_env.unwrapped
    obs, _ = env.reset()
    path = [(float(base.state["x"]), float(base.state["y"]))]
    path_options: list[int] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        option, _length = env._decode_action(int(action))
        n_measurements = len(gp_env.X)
        obs, _, terminated, truncated, _ = env.step(int(action))
        for position in gp_env.X[n_measurements:]:
            path.append((float(position[0]), float(position[1])))
            path_options.append(option)

    true_plume = gp_env.original_GT_c_over[:, :, 0].copy()
    true_border = gp_env.original_GT_c_around[:, :, 0].copy()
    visited = base.state["visited_map"][:, :, 0, 0].copy()
    gp_mean = np.array(gp_env.GP_pred, dtype=float)
    gp_std = np.array(gp_env.GP_std, dtype=float)
    results = map_metrics(gp_mean, true_plume, true_border, visited, gp_env.c_thresh, gp_env.c_std)
    results.update(
        ended_early=float(terminated),
        steps=float(env.primitive_steps),
        visited_share=float(np.mean(visited)),
        mean_gp_std=float(np.mean(gp_std)),
    )
    return EpisodeMaps(
        base.conc[:, :, 0].copy(),
        gp_mean,
        gp_std,
        true_plume,
        true_border,
        visited,
        np.array(path),
        np.array(path_options, dtype=int),
        float(gp_env.c_thresh),
        float(gp_env.c_std),
        results,
    )


def _draw_map(ax: plt.Axes, values: np.ndarray, **kwargs: Any) -> None:
    """Draw a map with x to the right and y up."""
    half = (values.shape[0] - 1) / 2
    ax.imshow(
        values.T,
        origin="lower",
        extent=(-half - 0.5, half + 0.5, -half - 0.5, half + 0.5),
        interpolation="nearest",
        **kwargs,
    )


def _draw_contours(ax: plt.Axes, values: np.ndarray, c_threshold: float, c_width: float) -> None:
    """The plume edge as a solid line and the edges of the border band as thin lines."""
    half = (values.shape[0] - 1) / 2
    coords = np.arange(-half, half + 1)
    for levels, color, width in (
        ([c_threshold], INK, 1.2),
        ([c_threshold - c_width, c_threshold + c_width], SECONDARY_INK, 0.5),
    ):
        reached = [level for level in levels if values.min() < level < values.max()]
        if reached:
            ax.contour(coords, coords, values.T, levels=reached, colors=color, linewidths=width)


def _draw_path(ax: plt.Axes, episode: EpisodeMaps) -> None:
    """Visited cells in gray, the true plume edge, and the path colored by the option that made each move."""
    _draw_map(ax, episode.visited.astype(float), cmap=ListedColormap([SURFACE, HAIRLINE]), vmin=0, vmax=1)
    _draw_contours(ax, episode.true_concentration, episode.c_threshold, episode.c_width)
    segments = np.stack([episode.path[:-1], episode.path[1:]], axis=1)
    colors = [OPTION_COLORS[option] for option in episode.path_options]
    ax.add_collection(LineCollection(list(segments), colors=colors, linewidths=1.3, capstyle="round"))
    ax.plot(*episode.path[0], "o", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=6, zorder=3)
    ax.plot(*episode.path[-1], "o", markerfacecolor=INK, markeredgecolor=SURFACE, markersize=6, zorder=3)


def _episode_label(label: str, episode: EpisodeMaps) -> str:
    results = episode.results
    ended = "ended by the goal" if results["ended_early"] else "budget used up"
    return (
        f"{label}\n{results['steps']:.0f} AUV steps\n{ended}\n"
        f"plume overlap {results['plume_overlap']:.3f}\nborder overlap {results['border_overlap']:.3f}\n"
        f"plumes found {results['plumes_found']:.0f} of {results['plumes']:.0f}\n"
        f"map visited {100 * results['visited_share']:.0f} %"
    )


def plot_comparison(episodes: list[tuple[str, EpisodeMaps]], first_option_name: str, output_dir: Path) -> Path:
    """One row per episode, one column per view. Saves map_comparison.png and .svg. Returns the path of the PNG."""
    rows = len(episodes)
    columns = len(PANEL_TITLES)
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.0 * columns + 2.0, 3.0 * rows + 1.5), squeeze=False, facecolor=SURFACE
    )
    bottom = 1.25 / (3.0 * rows + 1.5)
    fig.subplots_adjust(
        left=0.105, right=0.985, top=1 - 0.35 / (3.0 * rows + 1.5), bottom=bottom, wspace=0.08, hspace=0.08
    )
    concentration_scale = Normalize(0.0, 1.0)  # overlapping sources can go above 1: those cells show the darkest step
    uncertainty_scale = Normalize(0.0, 1.0)

    for row, (label, episode) in enumerate(episodes):
        plume, border = gp_maps(episode.gp_mean, episode.c_threshold, episode.c_width)
        row_axes = axes[row]
        _draw_path(row_axes[0], episode)
        _draw_map(row_axes[1], episode.true_concentration, cmap=CONCENTRATION_MAP, norm=concentration_scale)
        _draw_contours(row_axes[1], episode.true_concentration, episode.c_threshold, episode.c_width)
        _draw_map(row_axes[2], episode.gp_mean, cmap=CONCENTRATION_MAP, norm=concentration_scale)
        _draw_contours(row_axes[2], episode.gp_mean, episode.c_threshold, episode.c_width)
        _draw_map(
            row_axes[3], comparison_classes(episode.true_plume, plume), cmap=COMPARISON_MAP, vmin=NOT_GAS, vmax=EXTRA
        )
        _draw_map(
            row_axes[4], comparison_classes(episode.true_border, border), cmap=COMPARISON_MAP, vmin=NOT_GAS, vmax=EXTRA
        )
        _draw_map(row_axes[5], episode.gp_std, cmap=UNCERTAINTY_MAP, norm=uncertainty_scale)
        row_axes[0].annotate(
            _episode_label(label, episode),
            xy=(-0.16, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=9,
            color=INK,
        )

    half = (episodes[0][1].gp_mean.shape[0] - 1) / 2
    for row in range(rows):
        for column in range(columns):
            ax = axes[row, column]
            ax.set_facecolor(SURFACE)
            ax.set_xlim(-half - 0.5, half + 0.5)
            ax.set_ylim(-half - 0.5, half + 0.5)
            ax.set_xticks([-half, 0, half] if row == rows - 1 else [])
            ax.set_yticks([-half, 0, half] if column == 0 else [])
            ax.tick_params(colors=MUTED_INK, labelsize=8, length=2, color=NEUTRAL)
            for spine in ax.spines.values():
                spine.set_edgecolor(NEUTRAL)
                spine.set_linewidth(0.6)
            if row == 0:
                ax.set_title(PANEL_TITLES[column], fontsize=10, color=INK, pad=6)

    _add_legends(fig, axes[-1], first_option_name, concentration_scale, uncertainty_scale, bottom)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "map_comparison.svg", facecolor=SURFACE)
    png_path = output_dir / "map_comparison.png"
    fig.savefig(png_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return png_path


def _add_legends(
    fig: plt.Figure,
    last_row: np.ndarray,
    first_option_name: str,
    concentration_scale: Normalize,
    uncertainty_scale: Normalize,
    bottom: float,
) -> None:
    """Under the columns they belong to: the path and line legend, the two scales and the comparison legend."""
    legend_top = bottom - 0.32 / fig.get_figheight()
    path_handles = [
        Line2D([], [], color=OPTION_COLORS[0], lw=1.5, label=f"{first_option_name} moves"),
        Line2D([], [], color=OPTION_COLORS[1], lw=1.5, label="Plume agent moves"),
        Line2D([], [], color=OPTION_COLORS[2], lw=1.5, label="Border agent moves"),
        Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK, label="Start"),
        Line2D([], [], marker="o", ls="", markerfacecolor=INK, markeredgecolor=SURFACE, label="End"),
    ]
    line_handles = [
        Patch(facecolor=HAIRLINE, edgecolor=NEUTRAL, linewidth=0.5, label="Visited"),
        Line2D([], [], color=INK, lw=1.2, label="Plume edge"),
        Line2D([], [], color=SECONDARY_INK, lw=0.5, label="Border band edges"),
    ]
    comparison_handles = [
        Patch(facecolor=NEUTRAL, label="Correct"),
        Patch(facecolor=MISSED_COLOR, label="Missed: true gas, not in the estimate"),
        Patch(facecolor=EXTRA_COLOR, label="Extra: in the estimate, no true gas"),
    ]
    legend_style: dict[str, Any] = {
        "loc": "upper left",
        "frameon": False,
        "fontsize": 8,
        "labelcolor": SECONDARY_INK,
        "handlelength": 1.6,
        "borderaxespad": 0,
    }
    first = last_row[0].get_position()
    fig.legend(handles=path_handles, bbox_to_anchor=(0.005, legend_top), **legend_style)
    fig.legend(handles=line_handles, bbox_to_anchor=(first.x0, legend_top), **legend_style)
    comparison = last_row[3].get_position()
    fig.legend(handles=comparison_handles, bbox_to_anchor=(comparison.x0, legend_top), **legend_style)

    for columns, cmap, scale, label in (
        ((1, 2), CONCENTRATION_MAP, concentration_scale, "Concentration (0.3 is the plume threshold)"),
        ((5, 5), UNCERTAINTY_MAP, uncertainty_scale, "GP std"),
    ):
        left = last_row[columns[0]].get_position().x0
        right = last_row[columns[1]].get_position().x1
        height = 0.14 / fig.get_figheight()
        bar_axes = fig.add_axes((left, legend_top - height, right - left, height))
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=scale, cmap=cmap), cax=bar_axes, orientation="horizontal")
        bar.set_label(label, fontsize=8, color=SECONDARY_INK)
        bar.ax.tick_params(labelsize=8, colors=MUTED_INK, length=2, color=NEUTRAL)
        bar.outline.set_edgecolor(NEUTRAL)
        bar.outline.set_linewidth(0.6)


def save_results(episodes: list[EpisodeMaps], output_dir: Path) -> Path:
    """Save the results of every episode as map_comparison.csv. Returns its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "map_comparison.csv"
    keys = list(episodes[0].results)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", *keys])
        for number, episode in enumerate(episodes):
            writer.writerow([number, *(f"{episode.results[key]:.4f}" for key in keys)])
    return csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line of the evaluation script, plus the output folder."""
    parser = evaluation.make_parser()
    parser.description = "Compare the true map with the map the agent estimated, after an evaluation."
    parser.add_argument("--output", type=Path, required=True, help="folder for map_comparison.png, .svg and .csv")
    args = parser.parse_args(argv)
    if args.mode not in (None, "1", "4"):
        parser.error("--mode must be 1 (deterministic) or 4 (sampled, the default, as in training)")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    survey = evaluation.make_survey(args.model, **evaluation.survey_settings(args))
    deterministic = args.mode == "1"
    set_random_seed(evaluation.SEED)
    episodes = [run_episode(survey.env, survey.model, deterministic) for _ in range(args.episodes)]
    survey.env.close()

    evaluation.print_map_results([episode.results for episode in episodes])
    picked = pick_episodes([episode.score for episode in episodes])
    first_option_name = "Sweep" if survey.settings["spacing"] else "Space agent"
    png_path = plot_comparison([(label, episodes[index]) for label, index in picked], first_option_name, args.output)
    print(f"Saved {png_path}, {png_path.with_suffix('.svg')} and {save_results(episodes, args.output)}")


if __name__ == "__main__":
    main()
