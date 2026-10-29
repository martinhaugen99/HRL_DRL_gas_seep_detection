"""Tests for Option.value: the sub-agent's critic estimate, from its own observation keys only."""

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO

from HUGIN_gym.agents.options import Option


class DictEnv(gym.Env):
    """Minimal env with the dict observation of a sub-agent."""

    observation_space = gym.spaces.Dict(
        {
            "a": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "b": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }
    )
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        return self.observation_space.sample(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        return self.observation_space.sample(), 0.0, False, False, {}


def test_value_is_the_critic_estimate_of_the_filtered_observation() -> None:
    # Arrange
    model = PPO("MultiInputPolicy", DictEnv(), n_steps=8, batch_size=8, device="cpu", seed=0)
    option = Option(model, termination_fn=None, obs_keys=["a", "b"])
    sub_agent_obs = {"a": np.array([0.2, 0.7], dtype=np.float32), "b": np.array([0.4], dtype=np.float32)}
    meta_obs = {**sub_agent_obs, "not_for_this_sub_agent": np.zeros(3, dtype=np.float32)}

    # Act
    value = option.value(meta_obs)

    # Assert: same as the value of a normal forward pass
    obs_tensor, _ = model.policy.obs_to_tensor(sub_agent_obs)
    _, expected_value, _ = model.policy(obs_tensor)
    assert value == pytest.approx(expected_value.item(), rel=1e-5)
