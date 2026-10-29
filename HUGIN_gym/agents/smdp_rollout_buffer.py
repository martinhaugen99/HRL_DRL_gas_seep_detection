"""PPO rollout buffer with one discount per meta decision, for options that run different numbers of steps."""

import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictRolloutBuffer


class SMDPDictRolloutBuffer(DictRolloutBuffer):
    """DictRolloutBuffer that discounts each decision by its own factor (semi-MDP).

    An option that ran k steps of the base env is discounted by gamma**k (Sutton, Precup and Singh, 1999), where
    SB3 uses one gamma for every step. SMDPDiscountCallback writes the factors into `discounts` before each
    decision is added. Factors that are never written stay at the buffer's gamma, so with every factor equal to
    gamma the returns and advantages are those of DictRolloutBuffer.
    """

    discounts: np.ndarray

    def reset(self) -> None:
        super().reset()
        self.discounts = np.full((self.buffer_size, self.n_envs), self.gamma, dtype=np.float32)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """GAE(lambda) as in RolloutBuffer.compute_returns_and_advantage, with self.discounts[step] for gamma."""
        last_values_np = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values_np
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            discount = self.discounts[step]
            delta = self.rewards[step] + discount * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + discount * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values
