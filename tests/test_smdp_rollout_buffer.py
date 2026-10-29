"""Tests for SMDPDictRolloutBuffer: GAE(lambda) with one discount per meta decision."""

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer

from HUGIN_gym.agents.smdp_rollout_buffer import SMDPDictRolloutBuffer

GAMMA = 0.9
GAE_LAMBDA = 0.8
OBSERVATION_SPACE = spaces.Dict({"x": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)})
ACTION_SPACE = spaces.Discrete(2)


def make_buffer(buffer_class: type[DictRolloutBuffer], buffer_size: int, n_envs: int) -> DictRolloutBuffer:
    return buffer_class(
        buffer_size,
        OBSERVATION_SPACE,
        ACTION_SPACE,
        device="cpu",
        gae_lambda=GAE_LAMBDA,
        gamma=GAMMA,
        n_envs=n_envs,
    )


def fill(buffer: DictRolloutBuffer, rewards: np.ndarray, values: np.ndarray, episode_starts: np.ndarray) -> None:
    """Add one decision per row; one column per env."""
    n_envs = rewards.shape[1]
    for reward, value, episode_start in zip(rewards, values, episode_starts, strict=True):
        buffer.add(
            {"x": np.zeros((n_envs, 1), dtype=np.float32)},
            np.zeros((n_envs, 1)),
            reward.astype(np.float32),
            episode_start.astype(np.float32),
            th.tensor(value, dtype=th.float32),
            th.zeros(n_envs),
        )


def test_constant_discount_gives_the_sb3_returns_and_advantages() -> None:
    # Arrange
    rng = np.random.default_rng(0)
    rewards, values = rng.normal(size=(5, 2)), rng.normal(size=(5, 2))
    episode_starts = np.array([[1, 1], [0, 0], [0, 1], [0, 0], [1, 0]])
    last_values, dones = th.tensor([0.5, -0.2]), np.array([False, True])
    sb3_buffer = make_buffer(DictRolloutBuffer, 5, 2)
    smdp_buffer = make_buffer(SMDPDictRolloutBuffer, 5, 2)

    # Act
    for buffer in (sb3_buffer, smdp_buffer):
        fill(buffer, rewards, values, episode_starts)
        buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)

    # Assert: equal up to float32 rounding (SB3 multiplies gamma * gae_lambda as Python floats first)
    np.testing.assert_allclose(smdp_buffer.advantages, sb3_buffer.advantages, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(smdp_buffer.returns, sb3_buffer.returns, rtol=1e-5, atol=1e-6)


def test_each_decision_is_discounted_by_its_own_factor() -> None:
    # Arrange: a 2-step option, then a 5-step option, in one episode that goes on
    buffer = make_buffer(SMDPDictRolloutBuffer, 2, 1)
    fill(buffer, np.array([[1.0], [2.0]]), np.array([[0.5], [0.25]]), np.array([[1], [0]]))
    buffer.discounts[:, 0] = [GAMMA**2, GAMMA**5]

    # Act
    buffer.compute_returns_and_advantage(last_values=th.tensor([1.0]), dones=np.array([False]))

    # Assert
    delta_1 = 2.0 + GAMMA**5 * 1.0 - 0.25
    delta_0 = 1.0 + GAMMA**2 * 0.25 - 0.5
    advantage_0 = delta_0 + GAMMA**2 * GAE_LAMBDA * delta_1
    np.testing.assert_allclose(buffer.advantages[:, 0], [advantage_0, delta_1], rtol=1e-6)
    np.testing.assert_allclose(buffer.returns[:, 0], [advantage_0 + 0.5, delta_1 + 0.25], rtol=1e-6)


def test_reset_sets_every_discount_to_gamma() -> None:
    # Arrange
    buffer = make_buffer(SMDPDictRolloutBuffer, 3, 2)
    buffer.discounts[1] = 0.5

    # Act
    buffer.reset()

    # Assert
    np.testing.assert_allclose(buffer.discounts, np.full((3, 2), GAMMA), rtol=1e-6)
