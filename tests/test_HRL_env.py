"""Tests for the option loop of HierarchicalEnv: discounting inside an option, and full-length options."""

import gymnasium as gym
import numpy as np
import pytest

from HUGIN_gym.envs.HRL_env import HierarchicalEnv

GAMMA = 0.9


class StubBaseEnv:
    """Stand-in for the GP-wrapped HUGIN env: the AUV never makes progress, and the episode ends after end_step steps."""

    def __init__(self, end_step: int | None = None) -> None:
        empty_map = np.zeros((3, 3, 1, 1), dtype=bool)
        self.state = {
            key: empty_map.copy()
            for key in ("visited_map", "c_over_threshold_map", "c_around_threshold_map")
        }
        self.old_visited_sum = self.old_c_over_sum = self.old_c_around_sum = 0
        coverage_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {"plume_coverage": coverage_space, "border_coverage": coverage_space}
        )
        self.end_step = end_step
        self.steps = 0

    @property
    def unwrapped(self) -> "StubBaseEnv":
        return self

    def reset(self, **kwargs: object) -> tuple[dict, dict]:
        self.steps = 0
        return self._obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        self.steps += 1
        terminated = self.end_step is not None and self.steps >= self.end_step
        return self._obs(), 0.0, terminated, False, {}

    def _obs(self) -> dict:
        coverage = np.array([1.0], dtype=np.float32)  # plume and border covered
        return {"plume_coverage": coverage, "border_coverage": coverage}


class StubOption:
    def act(self, obs: dict) -> int:
        return 0


def make_env(max_opt_length: int, end_step: int | None = None) -> HierarchicalEnv:
    env = HierarchicalEnv(
        StubBaseEnv(end_step),
        [StubOption()],
        gamma=GAMMA,
        MAX_OPT_LENGTH=max_opt_length,
        ACCURACY_GOALS=[0.9, 0.9, 0.9],
        GP=True,
    )
    env.reset()
    return env


def test_option_rewards_are_discounted_inside_the_option() -> None:
    # Arrange
    env = make_env(max_opt_length=3)

    # Act
    _, reward, terminated, _, info = env.step(0)

    # Assert: -1 per step without new cells, discounted per step, and -2 for 3 steps without progress
    assert info["option_length"] == 3
    assert reward == pytest.approx(-(1 + GAMMA + GAMMA**2) - 2.0)
    assert not terminated


def test_option_ending_early_raises() -> None:
    # Arrange: 3 steps without progress end the option before its 5 steps
    env = make_env(max_opt_length=5)

    # Act and Assert
    with pytest.raises(RuntimeError, match="Option ended after 3 of 5 steps"):
        env.step(0)


def test_option_may_end_early_with_the_episode() -> None:
    # Arrange
    env = make_env(max_opt_length=5, end_step=2)

    # Act
    _, _, terminated, _, info = env.step(0)

    # Assert
    assert terminated
    assert info["option_length"] == 2
