import os
import pickle
import re
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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
        self.last_stats_file: Path | None = None # only the newest snapshot is kept, since each one holds all episodes so far
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
            self._save_snapshot()

        return True

    def _save_snapshot(self) -> None:
        """Write all episodes so far to <address>_<total steps>.pkl and remove the previous snapshot."""
        stats_file = Path(f"{self.address}_{self.num_timesteps}.pkl") # total steps, so a resumed run continues the numbering
        tmp_file = stats_file.with_name(f"{stats_file.name}.tmp")
        try: # write to a temporary file first, so a failed write never leaves a broken snapshot
            with tmp_file.open("wb") as f:
                pickle.dump(self.get_stats(), f)
        except OSError:
            tmp_file.unlink(missing_ok=True) # a partial snapshot is useless and only takes up disk quota
            raise
        os.replace(tmp_file, stats_file)
        # remove the previous snapshot only after the new one is written, so a crash always leaves one
        if self.last_stats_file is not None and self.last_stats_file != stats_file:
            self.last_stats_file.unlink(missing_ok=True)
        self.last_stats_file = stats_file

    def _snapshots(self) -> list[tuple[int, Path]]:
        """Step count and path of every <address>_<steps>.pkl snapshot."""
        address = Path(self.address)
        pattern = re.compile(rf"{re.escape(address.name)}_(\d+)\.pkl")
        return [
            (int(match.group(1)), path)
            for path in address.parent.glob(f"{address.name}_*.pkl")
            if (match := pattern.fullmatch(path.name))
        ]

    def load_snapshot(self, up_to_step: int) -> str:
        """Continue the stats from the newest readable snapshot, without the episodes after up_to_step.

        Call this when the model resumes from its checkpoint at up_to_step. The snapshot can be newer
        than the checkpoint (runs before 2026-10-07 saved stats every 5M steps and checkpoints every 10M),
        so the episodes are cut where their summed lengths pass up_to_step. The cut can keep up to one
        episode per env too many: those running at the checkpoint. The coverage lists skip some episodes,
        so they are cut at the same fraction of their length. Returns a summary for the resume log.
        """
        stats, loaded_file, unreadable = None, None, []
        for step, path in sorted(self._snapshots(), reverse=True):
            if step < up_to_step:
                break # older snapshots miss episodes before the checkpoint
            try:
                with path.open("rb") as f:
                    stats = pickle.load(f)
                loaded_file = path
                break
            except (EOFError, pickle.UnpicklingError): # cut short by a failed write, e.g. a full disk quota
                unreadable.append(path.name)
        if stats is None or loaded_file is None:
            raise FileNotFoundError(
                f"No readable {self.address}_<steps>.pkl at or after step {up_to_step} (unreadable: {unreadable})"
            )

        n_total = len(stats["episode_lengths"])
        n_keep = int(np.searchsorted(np.cumsum(stats["episode_lengths"]), up_to_step, side="right"))

        def cut_coverage(values: list) -> list:
            return values[: round(len(values) * n_keep / n_total)] if n_total else []

        self.episode_rewards = stats["episode_rewards"][:n_keep]
        self.episode_lengths = stats["episode_lengths"][:n_keep]
        self.episode_visited_count = stats["visited_states_counts"][:n_keep]
        self.episode_counter_above_threshold = cut_coverage(stats["episode_counter_above_threshold"])
        self.episode_counter_around_threshold = cut_coverage(stats["episode_counter_around_threshold"])
        if self.GP:
            self.ASSUMED_episode_counter_above_threshold = cut_coverage(stats["assumed_episode_counter_above_threshold"])
            self.ASSUMED_episode_counter_around_threshold = cut_coverage(stats["assumed_episode_counter_around_threshold"])
        self.last_stats_file = loaded_file # removed once the resumed run writes its first snapshot

        summary = f"stats from {loaded_file.name}: kept {n_keep} of {n_total} episodes (up to step {up_to_step})"
        if unreadable:
            summary += f"; skipped unreadable {', '.join(unreadable)}"
        return summary



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