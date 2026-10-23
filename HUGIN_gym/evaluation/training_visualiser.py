import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

training_stats_path =  "../../../trained-agents/plume_GT_faster"#HRL_3agents_Fixed_REW
suffix = "training_stats.pkl"#"episode_stats_15000000.pkl"

GP = True
file_path = os.path.join(os.path.dirname(__file__),training_stats_path, suffix)


with open(file_path, "rb") as f:
    stats = pickle.load(f)

# print(type(stats))
# print(stats.keys())
# print(type(stats["ASSUMED_episode_counter_around_threshold"]))
# print(len(stats["ASSUMED_episode_counter_around_threshold"]))
# exit()
episode_rewards = stats["episode_rewards"]  # list of list of rewards per step
episode_lengths = stats["episode_lengths"]  # list of lengths per episode
episode_visited_counts = stats["visited_states_counts"]  # visited states count per episode
N_plots = 4

if GP==False:
    if "episode_counter_around_threshold" in stats:
        episode_counter_around_threshold = stats["episode_counter_around_threshold"] # counter abouve thershold per episode
        N_plots+=1

    if "episode_counter_above_threshold" in stats:
        episode_counter_above_threshold = stats["episode_counter_above_threshold"] # counter abouve thershold per episode
        N_plots+=1
else:
    
    if "assumed_episode_counter_above_threshold" in stats:
        episode_counter_above_threshold = stats["assumed_episode_counter_above_threshold"] # counter abouve thershold per episode
        N_plots+=1
    if "assumed_episode_counter_around_threshold" in stats:
        episode_counter_around_threshold = stats["assumed_episode_counter_around_threshold"] # counter abouve thershold per episode
        N_plots+=1
   

def running_mean_std(data, window_size, step=10):
    means = []
    stds = []
    indices = []

    for i in range(len(data) // step):
        idx = i * step
        start_index = max(0, idx - window_size + 1)
        window = data[start_index:idx + 1]

        means.append(np.mean(window))
        stds.append(np.std(window))
        indices.append(idx)

    return indices, means, stds

total_rewards = [sum(rewards) for rewards in episode_rewards]

#running_avg_total_rewards = []
window_size = 50



idx_rewards, mean_rewards, std_rewards = running_mean_std(total_rewards, window_size)
idx_lengths, mean_lengths, std_lengths = running_mean_std(episode_lengths, window_size)
idx_visited, mean_visited, std_visited = running_mean_std(episode_visited_counts, window_size)

if GP==False:
    if "episode_counter_around_threshold" in stats:
        idx_counter, mean_counter, std_counter = running_mean_std(
            episode_counter_around_threshold, window_size
        )

    if "episode_counter_above_threshold" in stats:
        idx_counter2, mean_counter2, std_counter2 = running_mean_std(
            episode_counter_above_threshold, window_size
        )
else:
    if "assumed_episode_counter_around_threshold" in stats:
        idx_counter, mean_counter, std_counter = running_mean_std(
            episode_counter_around_threshold, window_size
        )

    if "assumed_episode_counter_above_threshold" in stats:
        idx_counter2, mean_counter2, std_counter2 = running_mean_std(
            episode_counter_above_threshold, window_size
        )

#----
# the_index = [i for i in range(len(total_rewards)) if i % 10 == 0][1:]  # indices corresponding to mean
# for i in range(len(total_rewards)//10):
#     start_index = max(0, i*10 - window_size + 1)
#     window = total_rewards[start_index:i*10 + 1]
#     running_avg = sum(window) / len(window)
#     running_avg_total_rewards.append(running_avg)

# running_avg_episode_lengths = []
# for i in range(len(episode_lengths)//10):
#     start_index = max(0, i*10 - window_size + 1)
#     window = episode_lengths[start_index:i*10 + 1]
#     running_avg = sum(window) / len(window)
#     running_avg_episode_lengths.append(running_avg)

# running_avg_visited_counts = []
# for i in range(len(episode_visited_counts)//10):
#     start_index = max(0, i*10 - window_size + 1)
#     window = episode_visited_counts[start_index:i*10 + 1]
#     running_avg = sum(window) / len(window)
#     running_avg_visited_counts.append(running_avg)

# running_avg_counter_above_threshold = []
# if "episode_counter_above_threshold" in stats:
#     for i in range(len(episode_counter_above_threshold)//10):
#         start_index = max(0, i*10 - window_size + 1)
#         window = episode_counter_above_threshold[start_index:i*10 + 1]
#         running_avg = sum(window) / len(window)
#         running_avg_counter_above_threshold.append(running_avg)

# -----

# finding the corresponding episode number to the desired waypoints
# waypoints_episode_indices = []
# if "waypoints" in stats:
#     waypoints = stats["waypoints"]
#     for wp in waypoints:
#         accumulated_steps = 0
#         episode_index = 0
#         for length in episode_lengths:
#             accumulated_steps += length
#             if accumulated_steps >= wp:
#                 break
#             episode_index += 1
#         waypoints_episode_indices.append(episode_index)

# Plotting
plt.figure(figsize=(5*N_plots, 5))

# Plot total reward per episode
plt.subplot(1, N_plots, 1)
#plt.plot(total_rewards[1:])

plt.plot(idx_rewards, mean_rewards, color="C1", linewidth=2)
plt.fill_between(
    idx_rewards,
    np.array(mean_rewards) - np.array(std_rewards),
    np.array(mean_rewards) + np.array(std_rewards),
    color="C1",
    alpha=0.25
)
# inserting a vertical line at the waypoints if available

# for wp in waypoints_episode_indices:
#     plt.axvline(x=wp, color='gray', linestyle='--', linewidth=0.8)  # Adjust x position for running average
plt.xlabel("Episode", fontsize=17)  # Increased fontsize
plt.ylabel("Total Reward", fontsize=17)  # Increased fontsize
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)  # Increased tick number size

# Plot episode lengths
plt.subplot(1, N_plots, 2)
#plt.plot(episode_lengths[1:], color="C3")
plt.plot(idx_lengths, mean_lengths, color="C3", linewidth=2)
plt.fill_between(
    idx_lengths,
    np.array(mean_lengths) - np.array(std_lengths),
    np.array(mean_lengths) + np.array(std_lengths),
    color="C3",
    alpha=0.25
)
plt.xlabel("Episode", fontsize=17)  # Increased fontsize
plt.ylabel("Episode Length (steps)", fontsize=17)  # Increased fontsize
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)  # Increased tick number size


# Plot reward curves per episode with color scale
plt.subplot(1, N_plots, 3)
cmap = plt.get_cmap('viridis')
N = len(episode_rewards)
step = 10  # Plot every 10th episode to reduce the number of lines
colors = [cmap(i / (N // step - 1)) for i in range(0, N, step)]

for i, (reward, color) in enumerate(zip(episode_rewards, colors)):
    plt.plot(reward, color=color)

plt.xlabel("Step within Episode", fontsize=17)  # Increased fontsize
plt.ylabel("Reward at Step", fontsize=17)  # Increased fontsize
#plt.ylim([0.1,-0.3])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)  # Increased tick number size

# Add a colorbar for episode number
norm = Normalize(vmin=1, vmax=N)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array for the colorbar
cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
cbar.set_label("Episode Number", fontsize=17)  # Increased fontsize

# Plot episode target distances
plt.subplot(1, N_plots, 4)
#plt.plot(episode_visited_counts[1:], color="C2")
mean_visited_arr = np.array(mean_visited)
std_visited_arr = np.array(std_visited)
# Clip uncertainty in [0, 1]
lower_visited = np.clip(mean_visited_arr - std_visited_arr, 0, 1)
upper_visited = np.clip(mean_visited_arr + std_visited_arr, 0, 1)
# convert to percentage
mean_visited_pct = mean_visited_arr * 100
lower_visited_pct = lower_visited * 100
upper_visited_pct = upper_visited * 100
#plt.plot(the_index, running_avg_counter_above_threshold, color="C4", linewidth=2)
plt.plot(idx_visited, mean_visited_pct, color="C2", linewidth=2)
plt.fill_between(
    idx_visited,
    lower_visited_pct,
    upper_visited_pct,
    color="C2",
    alpha=0.25
)
plt.xlabel("Episode", fontsize=17)  # Increased fontsize
plt.ylabel("Visited Positions (%)", fontsize=17)
plt.ylim(0, 100) # Increased fontsize
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)  # Increased tick number size

if "episode_counter_around_threshold" in stats:
    plt.subplot(1, N_plots, 5)
    mean_counter_arr = np.array(mean_counter)
    std_counter_arr = np.array(std_counter)

    lower_counter = np.clip(mean_counter_arr - std_counter_arr, 0, 1)
    upper_counter = np.clip(mean_counter_arr + std_counter_arr, 0, 1)

    mean_counter_pct = mean_counter_arr * 100
    lower_counter_pct = lower_counter * 100
    upper_counter_pct = upper_counter * 100

    plt.plot(idx_counter, mean_counter_pct, color="C4", linewidth=2)

    plt.fill_between(
        idx_counter,
        lower_counter_pct,
        upper_counter_pct,
        color="C4",
        alpha=0.25
    )

    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Steps Above Threshold (%)", fontsize=17)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=14)  # Increased tick number size


if "episode_counter_around_threshold" in stats:
    plt.subplot(1, N_plots, 5)
    mean_counter_arr = np.array(mean_counter)
    std_counter_arr = np.array(std_counter)

    lower_counter = np.clip(mean_counter_arr - std_counter_arr, 0, 1)
    upper_counter = np.clip(mean_counter_arr + std_counter_arr, 0, 1)

    mean_counter_pct = mean_counter_arr * 100
    lower_counter_pct = lower_counter * 100
    upper_counter_pct = upper_counter * 100

    plt.plot(idx_counter, mean_counter_pct, color="C4", linewidth=2)

    plt.fill_between(
        idx_counter,
        lower_counter_pct,
        upper_counter_pct,
        color="C4",
        alpha=0.25
    )

    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Steps Around Threshold (%)", fontsize=17)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=14)  # Increased tick number size

if "episode_counter_above_threshold" in stats:
    if N_plots==6:
        plt.subplot(1, N_plots, 6)
    else:
        plt.subplot(1, N_plots, 5)
    mean_counter_arr = np.array(mean_counter2)
    std_counter_arr = np.array(std_counter2)

    lower_counter = np.clip(mean_counter_arr - std_counter_arr, 0, 1)
    upper_counter = np.clip(mean_counter_arr + std_counter_arr, 0, 1)

    mean_counter_pct = mean_counter_arr * 100
    lower_counter_pct = lower_counter * 100
    upper_counter_pct = upper_counter * 100

    plt.plot(idx_counter2, mean_counter_pct, color="C4", linewidth=2)

    plt.fill_between(
        idx_counter2,
        lower_counter_pct,
        upper_counter_pct,
        color="C4",
        alpha=0.25
    )

    plt.xlabel("Episode", fontsize=17)
    plt.ylabel("Steps Above Threshold (%)", fontsize=17)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=14)  # Increased tick number size


# Remove title
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__),training_stats_path, "training_stats.svg"),format="svg")  # Save the figure with high resolution
plt.show()
