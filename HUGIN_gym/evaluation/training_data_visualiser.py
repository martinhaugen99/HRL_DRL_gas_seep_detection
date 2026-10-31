"""Plot the training stats of one or more runs.

Reads training_stats.pkl (written by EpisodeStatsCallback) from each run folder and saves
training_stats.png and training_stats.svg in the same folder.

Run from the repository root, e.g.:
    .venv/bin/python -m HUGIN_gym.evaluation.training_data_visualiser \
        ../trained-agents/HRL_PPO_*
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

WINDOW = 50  # episodes in the running mean and std
OPTIONAL_KEYS = {
    "Steps Around Threshold (%)": "episode_counter_around_threshold",
    "Steps Above Threshold (%)": "episode_counter_above_threshold",
}


# --- Helpers ---
def running_mean_std(data, window, step=10):
    idx, means, stds = [], [], []
    for i in range(0, len(data), step):
        w = data[max(0, i - window + 1) : i + 1]
        idx.append(i)
        means.append(np.mean(w))
        stds.append(np.std(w))
    return np.array(idx), np.array(means), np.array(stds)


def plot_with_std(ax, x, mean, std, color, ylabel, ylim=None, to_pct=False):
    if to_pct:
        mean, std = mean * 100, std * 100

    lower = mean - std
    upper = mean + std

    if to_pct:
        lower = np.clip(lower, 0, 100)
        upper = np.clip(upper, 0, 100)

    ax.plot(x, mean, color=color, lw=2)
    ax.fill_between(x, lower, upper, color=color, alpha=0.25)

    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True)
    ax.tick_params(labelsize=12)


def plot_reward_curves(ax, rewards):
    cmap = plt.get_cmap("viridis")
    step = 10  # this will make the plot production much faster
    N = len(rewards)

    for i in range(0, N, step):
        ax.plot(rewards[i], color=cmap(i / max(1, N - 1)))

    norm = Normalize(vmin=1, vmax=N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, pad=0.02).set_label("Episode Number", fontsize=14)

    ax.set_xlabel("Step within Episode", fontsize=14)
    ax.set_ylabel("Reward", fontsize=14)
    ax.grid(True)
    ax.tick_params(labelsize=12)


def plot_training_stats(run_dir: Path) -> Path:
    """Plot training_stats.pkl of a run folder and save the plot there as PNG and SVG.

    Returns the path of the PNG.
    """
    # --- Load ---
    with (run_dir / "training_stats.pkl").open("rb") as f:
        stats = pickle.load(f)

    # --- Data prep ---
    total_rewards = [sum(r) for r in stats["episode_rewards"]]

    data_map = {
        "Total Reward": total_rewards,
        "Episode Length": stats["episode_lengths"],
        "Visited Positions (%)": stats["visited_states_counts"],
    }

    # Optional metrics
    for label, key in OPTIONAL_KEYS.items():
        if key in stats:
            data_map[label] = stats[key]

    # --- Plot ---
    N_plots = len(data_map) + 1  # + reward curves
    fig, axes = plt.subplots(1, N_plots, figsize=(5 * N_plots, 5))
    axes = np.atleast_1d(axes)

    # --- Main plots ---
    for ax, (label, data), color in zip(
        axes, data_map.items(), ["C1", "C3", "C2", "C4", "C5"]
    ):
        x, m, s = running_mean_std(data, WINDOW)

        plot_with_std(
            ax,
            x,
            m,
            s,
            color,
            ylabel=label,
            ylim=(0, 100) if "%" in label else None,
            to_pct="%" in label,
        )

    # --- Reward curves (last subplot) ---
    plot_reward_curves(axes[-1], stats["episode_rewards"])

    # --- Save ---
    fig.tight_layout()
    fig.savefig(run_dir / "training_stats.svg")
    png_path = run_dir / "training_stats.png"
    fig.savefig(png_path)
    plt.close(fig)  # free the memory before the next run
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the training stats of run folders as training_stats.png/.svg."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="run folders, e.g. ../trained-agents/HRL_PPO_minimal_4371279",
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        print(f"Saved {plot_training_stats(run_dir)}")


if __name__ == "__main__":
    main()
