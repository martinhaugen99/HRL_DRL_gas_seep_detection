from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pickle
import os

class EpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose=0, save_freq=20_000, max_episode_length=121, NUM_ENVS=1,address=None, N_states=0,GP=False):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []
        #self.episode_velocities = []
        self.episode_counter = 0
        self.episode_visited_count = []
        self.episode_coverage_count =[]
        self.save_freq = save_freq
        self.max_episode_length = max_episode_length
        self.NUM_ENVS = NUM_ENVS
        self.address = address
        self.last_stats_file = None # only the newest snapshot is kept, since each one holds all episodes so far
        #self.N_states=N_states
        self.episode_counter_above_threshold = []
        self.episode_counter_around_threshold = []

        # Per-env running episode data (initialized in _on_training_start)
        self.current_rewards = None
        self.current_lengths = None
        self.GP = GP
        if GP:
            self.ASSUMED_episode_counter_above_threshold = []
            self.ASSUMED_episode_counter_around_threshold = []
    def _on_training_start(self) -> None:
        n_envs = self.NUM_ENVS

        self.current_rewards = [[] for _ in range(n_envs)]
        self.current_lengths = [0 for _ in range(n_envs)]
        #self.current_above_threshold = [0 for _ in range(n_envs)]
    def _on_step(self) -> bool:

        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for env_idx in range(len(dones)):
            # Accumulate per-env episode data
            self.current_rewards[env_idx].append(rewards[env_idx])
            self.current_lengths[env_idx] += 1
            
            

            if dones[env_idx]:
                # Episode finished in this env
                ep_rewards = self.current_rewards[env_idx]
                ep_length = self.current_lengths[env_idx]

                self.episode_rewards.append(ep_rewards.copy())
                self.episode_lengths.append(ep_length)

                visited_count = infos[env_idx].get("visited_states_count", None)
                self.episode_visited_count.append(visited_count)
                if self.GP==True:
                    if "actual_current_above_threshold" in infos[env_idx]:
                        self.episode_counter_above_threshold.append(infos[env_idx]["actual_current_above_threshold"]) # normed in bluerov env already
                    if "actual_current_around_threshold" in infos[env_idx]:
                        self.episode_counter_around_threshold.append(infos[env_idx]["actual_current_around_threshold"])
                    if "assumed_current_above_threshold" in infos[env_idx]:
                        self.ASSUMED_episode_counter_above_threshold.append(infos[env_idx]["assumed_current_above_threshold"]) # normed in bluerov env already
                    if "assumed_current_around_threshold" in infos[env_idx]:
                        self.ASSUMED_episode_counter_around_threshold.append(infos[env_idx]["assumed_current_around_threshold"])
                else:
                    if "current_above_threshold" in infos[env_idx]:
                        self.episode_counter_above_threshold.append(infos[env_idx]["current_above_threshold"]) # normed in bluerov env already
                    if "current_around_threshold" in infos[env_idx]:
                        self.episode_counter_around_threshold.append(infos[env_idx]["current_around_threshold"])

                # Reset this env's buffers
                self.current_rewards[env_idx].clear()
                self.current_lengths[env_idx] = 0
                #self.current_above_threshold[env_idx] = 0

        # Save periodically (callback calls, not raw timesteps)
        if self.n_calls % self.save_freq == 0:
            stats=self.get_stats()
            stats_file = f"{self.address}_{self.n_calls*self.NUM_ENVS}.pkl"
            with open(stats_file, "wb") as f:
                pickle.dump(stats, f)
            # remove the previous snapshot only after the new one is written, so a crash always leaves one
            if self.last_stats_file is not None and os.path.exists(self.last_stats_file):
                os.remove(self.last_stats_file)
            self.last_stats_file = stats_file

        return True
    
    

    def get_stats(self):
        dict_to_return = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "visited_states_counts": self.episode_visited_count,
            "episode_counter_above_threshold": self.episode_counter_above_threshold,
            "episode_counter_around_threshold": self.episode_counter_around_threshold
        }

        if self.GP:
            dict_to_return["assumed_episode_counter_above_threshold"] = self.ASSUMED_episode_counter_above_threshold
            dict_to_return["assumed_episode_counter_around_threshold"] = self.ASSUMED_episode_counter_around_threshold
        return dict_to_return