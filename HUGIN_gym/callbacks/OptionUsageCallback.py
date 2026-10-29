import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class OptionUsageCallback(BaseCallback):
    """Logs to TensorBoard which actions the meta agent picks in each third of the survey, once per rollout.

    Reads the info that HierarchicalEnv adds to every decision (survey_fraction, option_length, survey_steps).
    Also logs the mean option length, and for the episodes that ended in the rollout their length in base env
    steps and the share that reached the goal before the survey budget ran out.
    """

    PHASES = ("early", "middle", "late")

    def __init__(self, action_names: list[str], verbose: int = 0) -> None:
        super().__init__(verbose)
        self.action_names = action_names
        self._clear()

    def _clear(self) -> None:
        self.choices = np.zeros((len(self.PHASES), len(self.action_names)), dtype=np.int64)
        self.option_lengths: list[int] = []
        self.episode_steps: list[int] = []
        self.episode_ended_early: list[bool] = []

    def _on_step(self) -> bool:
        for action, info, done in zip(self.locals["actions"], self.locals["infos"], self.locals["dones"], strict=True):
            phase = min(int(info["survey_fraction"] * len(self.PHASES)), len(self.PHASES) - 1)
            self.choices[phase, int(action)] += 1
            self.option_lengths.append(info["option_length"])
            if done:
                self.episode_steps.append(info["survey_steps"])
                self.episode_ended_early.append(not info.get("TimeLimit.truncated", False))
        return True

    def _on_rollout_end(self) -> None:
        shares = self.choices / np.maximum(self.choices.sum(axis=1, keepdims=True), 1)
        for phase_idx, phase in enumerate(self.PHASES):
            for action_idx, name in enumerate(self.action_names):
                self.logger.record(f"hrl_choice_{phase}/{name}", shares[phase_idx, action_idx])
        if self.option_lengths:
            self.logger.record("hrl/option_length", np.mean(self.option_lengths))
        if self.episode_steps:
            self.logger.record("hrl/episode_steps", np.mean(self.episode_steps))
            self.logger.record("hrl/ended_early", np.mean(self.episode_ended_early))
        self._clear()
