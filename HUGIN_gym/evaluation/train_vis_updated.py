import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from HUGIN_gym.utils.io import load_training_config_from_model
# =========================
# CONFIG
# =========================
#training_stats_path = "../../../trained-agents/NEW_plume_GP_clipped_RWD_min_steps_in_plume"
#training_stats_path = "../../../trained-agents/HRL_PPO_GP_base_agents_GP_1G_term_0p1_uncertainty_AND_all_plume_covered_3_OPT_len_90prosent_d_base_agents_only_subtask_termination_correctHRL_feature_extr"
#training_stats_path = "../../../trained-agents/SAC_space_GT_not_clipped_RWD"
#training_stats_path = "../../../trained_agents/DQN_gp_230steps_4218364" # DQN
training_stats_path = "../../../trained_agents/PPO_gp_230steps_4218364" # PPO
#training_stats_path = "../../../trained_agents/SAC_gp_230steps_4218364" # SAC
suffix = "episode_stats_9999960.pkl"#training_stats.pkl"


GP = True
SUB_POLICY_GP = True
AGENT_TYPE = "PLUME"

window_size = 500
step = 10

# =========================
# LOAD DATA
# =========================
file_path = os.path.join(os.path.dirname(__file__), training_stats_path, suffix)
#print(file_path)
config = load_training_config_from_model(file_path)
max_eps_len = config["ENVIRONMENT"]["max_episode_length"]
with open(file_path, "rb") as f:
    stats = pickle.load(f)

print("Available keys in stats:")
for k in stats.keys():
    print(k)

episode_rewards = stats["episode_rewards"]
episode_lengths = stats["episode_lengths"]
episode_visited_counts = stats["visited_states_counts"]

total_rewards = [sum(r) for r in episode_rewards]

# =========================
# METRIC SELECTION (FIXED)
# =========================
metrics = {
    "reward": total_rewards,
    "length": episode_lengths,
    "visited": episode_visited_counts,
}

if not GP:
    if "episode_counter_around_threshold" in stats:
        metrics["around"] = stats["episode_counter_around_threshold"]

    if "episode_counter_above_threshold" in stats:
        metrics["above"] = stats["episode_counter_above_threshold"]

else:
    if AGENT_TYPE == "PLUME":

        # REAL
        if "episode_counter_above_threshold" in stats:
            metrics["above"] = stats["episode_counter_above_threshold"]

        # ASSUMED (add ON TOP, not instead)
        if SUB_POLICY_GP and "assumed_episode_counter_above_threshold" in stats:
            metrics["assumed_above"] = stats["assumed_episode_counter_above_threshold"]

    elif AGENT_TYPE == "BORDER":

        # REAL
        if "episode_counter_around_threshold" in stats:
            metrics["around"] = stats["episode_counter_around_threshold"]

        # ASSUMED
        if SUB_POLICY_GP and "assumed_episode_counter_around_threshold" in stats:
            metrics["assumed_around"] = stats["assumed_episode_counter_around_threshold"]

    else:
        # fallback: include everything available
        if "episode_counter_around_threshold" in stats:
            metrics["around"] = stats["episode_counter_around_threshold"]
        if "episode_counter_above_threshold" in stats:
            metrics["above"] = stats["episode_counter_above_threshold"]

        if "assumed_episode_counter_around_threshold" in stats:
            metrics["assumed_around"] = stats["assumed_episode_counter_around_threshold"]
        if "assumed_episode_counter_above_threshold" in stats:
            metrics["assumed_above"] = stats["assumed_episode_counter_above_threshold"]

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

    return indices, np.array(means), np.array(stds)

running_stats = {}
for key, data in metrics.items():
    running_stats[key] = running_mean_std(data, window_size, step)

# =========================
# PLOT CONFIG
# =========================
plot_config = {
    "reward": ("C1", "Total Reward", False,False),
    "length": ("C3", "Episode Length (steps)", False, True),
    "visited": ("C2", "Visited Positions (%)", True),

    "around": ("C4", "Steps Around Threshold (%)", True),
    "above": ("C5", "Steps Above Threshold (%)", True),

    "assumed_around": ("C6", "Assumed Around Threshold (%)", True),
    "assumed_above": ("C7", "Assumed Above Threshold (%)", True),
}

def plot_with_std(ax, idx, mean, std, color, ylabel, percent=False,clip = False):
    if percent:
        lower = np.clip(mean - std, 0, 1) * 100
        upper = np.clip(mean + std, 0, 1) * 100
        mean = mean * 100
    elif clip:
        lower = np.clip(mean - std, 0, max_eps_len) 
        upper = np.clip(mean + std, 0, max_eps_len) 
    else:
        lower = mean - std
        upper = mean + std

    ax.plot(idx, mean, color=color, linewidth=2)
    ax.fill_between(idx, lower, upper, color=color, alpha=0.25)

    ax.set_xlabel("Episode", fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=14)

# =========================
# PLOTTING
# =========================
N_plots = len(metrics) + 1  # +1 for reward curves

plt.figure(figsize=(5 * N_plots, 5))
plot_idx = 1

# --- Standard metrics ---
for key in ["reward", "length"]:
    if key in running_stats:
        ax = plt.subplot(1, N_plots, plot_idx)
        idx, mean, std = running_stats[key]
        color, ylabel, percent, clip = plot_config[key]
        plot_with_std(ax, idx, mean, std, color, ylabel, percent, clip=clip)
        plot_idx += 1

# --- Reward curves ---
ax = plt.subplot(1, N_plots, plot_idx)
cmap = plt.get_cmap('viridis')
N = len(episode_rewards)

colors = [cmap(i / (N // step - 1)) for i in range(0, N, step)]

for reward, color in zip(episode_rewards[::step], colors):
    ax.plot(reward, color=color)

ax.set_xlabel("Step within Episode", fontsize=17)
ax.set_ylabel("Reward at Step", fontsize=17)
ax.grid(True)
ax.tick_params(axis='both', labelsize=14)

# Colorbar
norm = Normalize(vmin=1, vmax=N)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Episode Number", fontsize=17)

plot_idx += 1

# --- Remaining metrics ---
for key in ["visited", "around", "above","assumed_around","assumed_above"]:
    if key in running_stats:
        ax = plt.subplot(1, N_plots, plot_idx)
        idx, mean, std = running_stats[key]
        color, ylabel, percent = plot_config[key]
        plot_with_std(ax, idx, mean, std, color, ylabel, percent)
        plot_idx += 1

# =========================
# SAVE + SHOW
# =========================
plt.tight_layout()

save_path = os.path.join(
    os.path.dirname(__file__),
    training_stats_path,
    "training_stats.png"
)

plt.savefig(save_path, format="png")
plt.show()