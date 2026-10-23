import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

from HUGIN_gym.utils.io import load_training_config_from_model

# =========================
# CONFIG
# =========================

base_dir = "../../../trained-agents"

runs = [
    "DDQN_space",
    "paper_2D_PPO_space_10M",
    "SAC_space_GT_not_clipped_RWD",
    # "3D_PPO_space_test",  # 3D env + only ~300k steps -> not comparable to the 2D runs above
]

suffix = "training_stats.pkl"

# fixed colors per run
run_colors = {
    "DDQN_space":                    "green",
    "paper_2D_PPO_space_10M":        "violet",
    "SAC_space_GT_not_clipped_RWD":  "red",
}

# nice labels per run (legend names)
run_labels = {
    "DDQN_space":                    "(a) DDQN",
    "paper_2D_PPO_space_10M":        "(b) PPO",
    "SAC_space_GT_not_clipped_RWD":  "(c) SAC",
}

window_size = 20000
step = 10

GP = False         # SPACE agent runs are not GP/plume/border-based
SUB_POLICY_GP = False
AGENT_TYPE = "SPACE"

# Which metrics to plot as separate subplots
metrics_to_plot = ["reward", "length", "visited"]


# =========================
# RUNNING MEAN / STD
# =========================
def running_mean_std(data, window_size, step=10):
    means, stds, indices = [], [], []
    for i in range(len(data) // step):
        idx = i * step
        start = max(0, idx - window_size + 1)
        window = data[start:idx + 1]
        means.append(np.mean(window))
        stds.append(np.std(window))
        indices.append(idx)
    return np.array(indices), np.array(means), np.array(stds)


# =========================
# LOAD ALL RUNS
# =========================
all_running_stats = {}      # {run_name: {metric: (idx, mean, std)}}
max_eps_len_per_run = {}    # {run_name: max_episode_length}

for run in runs:
    training_stats_path = os.path.join(os.path.dirname(__file__),base_dir, run)
    file_path = os.path.join(training_stats_path, suffix)

    config = load_training_config_from_model(file_path)
    max_eps_len = config["ENVIRONMENT"]["max_episode_length"]
    max_eps_len_per_run[run] = max_eps_len

    with open(file_path, "rb") as f:
        stats = pickle.load(f)

    # Basic episode statistics
    episode_rewards = stats["episode_rewards"]          # list of list of rewards
    episode_lengths = stats["episode_lengths"]          # list of ints
    episode_visited_counts = stats["visited_states_counts"]  # list of something

    total_rewards = [sum(r) for r in episode_rewards]

    metrics = {
        "reward":  total_rewards,
        "length":  episode_lengths,
        "visited": episode_visited_counts,
    }

    # Optional GP / Meta-GP metrics (available in stats when GP / SUB_POLICY_GP)
    if not GP:
        if "episode_counter_around_threshold" in stats:
            metrics["around"] = stats["episode_counter_around_threshold"]
        if "episode_counter_above_threshold" in stats:
            metrics["above"] = stats["episode_counter_above_threshold"]
    else:
        if AGENT_TYPE == "PLUME":
            if "episode_counter_above_threshold" in stats:
                metrics["above"] = stats["episode_counter_above_threshold"]
            if SUB_POLICY_GP and "assumed_episode_counter_above_threshold" in stats:
                metrics["assumed_above"] = stats["assumed_episode_counter_above_threshold"]
        elif AGENT_TYPE == "BORDER":
            if "episode_counter_around_threshold" in stats:
                metrics["around"] = stats["episode_counter_around_threshold"]
            if SUB_POLICY_GP and "assumed_episode_counter_around_threshold" in stats:
                metrics["assumed_around"] = stats["assumed_episode_counter_around_threshold"]
        else:
            if "episode_counter_around_threshold" in stats:
                metrics["around"] = stats["episode_counter_around_threshold"]
            if "episode_counter_above_threshold" in stats:
                metrics["above"] = stats["episode_counter_above_threshold"]
            if "assumed_episode_counter_around_threshold" in stats:
                metrics["assumed_around"] = stats["assumed_episode_counter_around_threshold"]
            if "assumed_episode_counter_above_threshold" in stats:
                metrics["assumed_above"] = stats["assumed_episode_counter_above_threshold"]

    run_stats = {}
    for key, data in metrics.items():
        run_stats[key] = running_mean_std(data, window_size, step)

    all_running_stats[run] = run_stats


# =========================
# PLOT CONFIG
# =========================
# label, percent?, clip_to_max_eps_len?
plot_config = {
    "reward":  ("Total Reward", False, False),
    "length":  ("Episode Length [steps]", False, True),
    "visited": ("Visited Positions [%]", True, False),

    # If you later add:
    # "around": ("Steps Around Threshold (%)", True, False),
    "above": (r"visited $\wedge$ (GT:$c>c_{t}$) [%]", True, False),
    # "assumed_around": ("Assumed Around Threshold (%)", True, False),
    "assumed_above": (r"visited $\wedge$ (GP:$c>c_{t}$) [%]", True, False),
}


def plot_with_std(
    ax, idx, mean, std, color, ylabel,
    percent=False, clip=False, max_eps_len=None,
    label=None
):
    # mean / std transformation
    if percent:
        lower = np.clip(mean - std, 0, 1) * 100
        upper = np.clip(mean + std, 0, 1) * 100
        mean = mean * 100
    elif clip and max_eps_len is not None:
        lower = np.clip(mean - std, 0, max_eps_len)
        upper = np.clip(mean + std, 0, max_eps_len)
    else:
        lower = mean - std
        upper = mean + std

    # line with label -> legend uses this
    ax.plot(idx, mean, color=color, linewidth=2, label=label)

    # std band without label
    ax.fill_between(idx, lower, upper, color=color, alpha=0.25)

    # axis styling
    ax.set_xlabel("Episode", fontsize=22, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=22, fontweight="bold")
    ax.grid(True)
    ax.tick_params(axis="both", labelsize=18)


# =========================
# PLOTTING: ONE FIGURE, 3 SUBPLOTS
# =========================
N_plots = len(metrics_to_plot)
fig, axes = plt.subplots(1, N_plots, figsize=(5 * N_plots, 5), squeeze=False,constrained_layout = True)
axes = axes[0]  # flatten the row

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    ylabel, percent, clip = plot_config[metric]

    for run in runs:
        if metric not in all_running_stats[run]:
            continue

        idx, mean, std = all_running_stats[run][metric]
        color = run_colors[run]
        label = run_labels.get(run, run)  # fallback to folder name if missing
        max_eps_len = max_eps_len_per_run[run]

        plot_with_std(
            ax, idx, mean, std, color,
            ylabel,
            percent=percent,
            clip=clip,
            max_eps_len=max_eps_len,
            label=label
        )

    #ax.set_title(metric, fontsize=18, fontweight="bold")

    # Build legend from plotted handles; remove duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

handles, labels = axes[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))
fig.legend(
    unique.values(),
    unique.keys(),
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),   # to the right of the plots
    borderaxespad=0.0,
    fontsize=15,
    frameon=False,                # or True if you want a box
)


plt.tight_layout()

save_path = os.path.join(
    os.path.dirname(__file__),
    base_dir,
    "comparison_RL_algs.pdf"
)
plt.savefig(save_path, format="pdf",bbox_inches="tight")
plt.show()
