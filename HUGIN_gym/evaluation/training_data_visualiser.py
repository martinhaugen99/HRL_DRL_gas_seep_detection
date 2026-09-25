import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# --- Load ---
#base_path = os.path.join(os.path.dirname(__file__), "../../../trained_agents/PPO_gp_230steps_4218364")
#base_path = os.path.join(os.path.dirname(__file__), "../../../trained_agents/SAC_gp_230steps_4218364")
base_path = os.path.join(os.path.dirname(__file__), "../../../trained_agents/DQN_gp_230steps_4218364")

with open(os.path.join(base_path, "training_stats.pkl"), "rb") as f:
    stats = pickle.load(f)

# --- Helpers ---
def running_mean_std(data, window, step=10):
    idx, means, stds = [], [], []
    for i in range(0, len(data), step):
        w = data[max(0, i - window + 1): i + 1]
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
    step = 10 # this will make the plot production much faster
    N = len(rewards)

    for i in range(0, N, step):
        ax.plot(rewards[i], color=cmap(i / max(1, N - 1)))

    norm = Normalize(vmin=1, vmax=N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.02).set_label("Episode Number", fontsize=14)

    ax.set_xlabel("Step within Episode", fontsize=14)
    ax.set_ylabel("Reward", fontsize=14)
    ax.grid(True)
    ax.tick_params(labelsize=12)


# --- Data prep ---
window = 50

total_rewards = [sum(r) for r in stats["episode_rewards"]]

data_map = {
    "Total Reward": total_rewards,
    "Episode Length": stats["episode_lengths"],
    "Visited Positions (%)": stats["visited_states_counts"],
}

# Optional metrics
optional_keys = {
    "Steps Around Threshold (%)": "episode_counter_around_threshold",
    "Steps Above Threshold (%)": "episode_counter_above_threshold",
}

for label, key in optional_keys.items():
    if key in stats:
        data_map[label] = stats[key]

# --- Plot ---
N_plots = len(data_map) + 1  # + reward curves
fig, axes = plt.subplots(1, N_plots, figsize=(5 * N_plots, 5))
axes = np.atleast_1d(axes)

# --- Main plots ---
for ax, (label, data), color in zip(
    axes,
    data_map.items(),
    ["C1", "C3", "C2", "C4", "C5"]
):
    x, m, s = running_mean_std(data, window)

    plot_with_std(
        ax,
        x,
        m,
        s,
        color,
        ylabel=label,
        ylim=(0, 100) if "%" in label else None,
        to_pct="%" in label
    )

# --- Reward curves (last subplot) ---
plot_reward_curves(axes[-1], stats["episode_rewards"])

# --- Save ---
plt.tight_layout()
figpath = os.path.join(base_path, "training_stats.svg")
plt.savefig(figpath)
os.system(f"display {figpath}")
#plt.show()