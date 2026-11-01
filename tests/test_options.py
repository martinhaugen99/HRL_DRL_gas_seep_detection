"""Tests for Option.value (the sub-agent's critic estimate, from its own observation keys only) and SweepOption."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO

from HUGIN_gym.agents.options import (
    FORWARD,
    LEFT,
    QUARTER_TURN,
    RIGHT,
    Option,
    SweepOption,
)
from HUGIN_gym.envs.HRL_env import HierarchicalEnv
from HUGIN_gym.envs.HUGIN_env import HUGIN
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper

BOUNDARY_PENALTY = -4.0
# the survey starts of HUGIN._spawn_at_survey_start: a corner, heading along one of its two edges into the map
SURVEY_STARTS = [
    (-19, -19, 0),
    (-19, -19, 1),
    (19, -19, 2),
    (19, -19, 1),
    (-19, 19, 0),
    (-19, 19, 3),
    (19, 19, 2),
    (19, 19, 3),
]


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


def make_sweep(spacing: int, x: int = -19, y: int = -19, heading: int = 1) -> tuple[HUGIN, SimpleNamespace, SweepOption]:
    """A sweep on the real base env, with a GP that knows nothing yet."""
    base = HUGIN()
    base.HRL = True  # the base env gives no reward except the boundary penalty
    base.reset()
    base.state["x"], base.state["y"], base.state["theta"] = x, y, heading * QUARTER_TURN
    gp = SimpleNamespace(GP_std=np.ones((41, 41)))
    option = SweepOption(base, gp, spacing)
    option.reset()
    return base, gp, option


@pytest.mark.parametrize("heading", range(4))
@pytest.mark.parametrize("action", [FORWARD, LEFT, RIGHT])
def test_moves_are_those_of_the_env(heading: int, action: int) -> None:
    # Arrange
    base, _, option = make_sweep(spacing=4, x=0, y=0, heading=heading)
    expected = next(move for move in option.moves[heading] if move[0] == action)

    # Act
    base.step(action)

    # Assert
    assert option._pose() == (expected[1], expected[2], expected[3])


def test_legal_poses_are_those_the_env_accepts() -> None:
    # Arrange
    base, _, option = make_sweep(spacing=4)

    # Act and Assert
    for x in range(-21, 22):
        for y in range(-21, 22):
            for heading in range(4):
                rejected = base._check_boundary_violation(x, y, 0, heading * QUARTER_TURN)
                assert option._is_legal(x, y, heading) != rejected, (x, y, heading)


@pytest.mark.parametrize(("x", "y", "heading"), SURVEY_STARTS)
@pytest.mark.parametrize(("spacing", "max_steps", "min_visited"), [(2, 430, 0.95), (4, 230, 0.70), (6, 165, 0.50)])
def test_sweep_covers_the_map_without_hitting_the_edge(x: int, y: int, heading: int, spacing: int, max_steps: int, min_visited: float) -> None:
    # Arrange
    base, _, option = make_sweep(spacing, x, y, heading)

    # Act
    steps = 0
    while option.waypoints and steps < 500:
        _, reward, *_ = base.step(option.act({}))
        assert reward != BOUNDARY_PENALTY
        steps += 1
        option.act({})  # removes the waypoint just reached

    # Assert
    assert not option.waypoints
    assert steps <= max_steps
    assert np.mean(base.state["visited_map"]) >= min_visited


def test_sweep_resumes_at_the_next_waypoint_after_another_option_moved_the_auv() -> None:
    # Arrange
    base, _, option = make_sweep(spacing=4)
    next_waypoint = option.waypoints[0]

    # Act: another option moved the AUV across the map
    base.state["x"], base.state["y"], base.state["theta"] = 15, 15, 0.0
    for _ in range(60):
        base.step(option.act({}))
        if abs(base.state["x"] - next_waypoint[0]) <= 1 and abs(base.state["y"] - next_waypoint[1]) <= 1:
            break

    # Assert
    assert abs(base.state["x"] - next_waypoint[0]) <= 1
    assert abs(base.state["y"] - next_waypoint[1]) <= 1


def test_sweep_skips_waypoints_the_gp_already_knows() -> None:
    # Arrange
    _, gp, option = make_sweep(spacing=4)
    known = [option.waypoints[i] for i in range(2, 6)]  # the AUV stands on the first ones
    n_waypoints = len(option.waypoints)
    for x, y in known:
        gp.GP_std[x + 20, y + 20] = 0.0

    # Act: reach the waypoints before the known ones, then ask for the next action
    option.waypoints.popleft()
    option.waypoints.popleft()
    option.act({})

    # Assert
    assert len(option.waypoints) == n_waypoints - 6


def test_after_the_sweep_the_auv_goes_to_the_least_known_cells() -> None:
    # Arrange
    base, gp, option = make_sweep(spacing=4, x=-19, y=-19, heading=0)
    option.waypoints.clear()
    gp.GP_std = np.zeros((41, 41))
    gp.GP_std[30:35, 30:35] = 1.0  # the least known part, more than 1 % of the map

    # Act
    for _ in range(40):
        base.step(option.act({}))
        if base.state["x"] + 20 in range(29, 36) and base.state["y"] + 20 in range(29, 36):
            break

    # Assert: within 1 cell of it
    assert base.state["x"] + 20 in range(29, 36)
    assert base.state["y"] + 20 in range(29, 36)


def test_spacing_must_be_positive() -> None:
    # Act and Assert
    with pytest.raises(ValueError, match="spacing"):
        SweepOption(HUGIN(), SimpleNamespace(GP_std=np.ones((41, 41))), spacing=0)


def test_value_of_the_sweep_is_zero() -> None:
    # Arrange
    _, _, option = make_sweep(spacing=4)

    # Act and Assert
    assert option.value({}) == 0.0


def test_always_choosing_the_sweep_is_a_lawnmower_that_reaches_the_visit_goal() -> None:
    # Arrange: the real env of the HRL evaluation, with tracks 2 cells apart
    base = HUGIN()
    base.train = True
    base.use_c_map = True
    base.GP_ON = True
    base.HRL = True
    base.agent_type = "META"  # type: ignore[attr-defined]  # set by the scripts, not in HUGIN.__init__
    base.spawn_at_survey_start = True
    base.vary_plume_size = True
    gp_env = GPWrapper(base, HRL=True)
    env = HierarchicalEnv(
        gp_env,
        [SweepOption(base, gp_env, spacing=2)],
        gamma=0.998,
        MAX_OPT_LENGTH=3,
        ACCURACY_GOALS=[0.9, 0.9, 0.9],
        GP=True,
        max_primitive_steps=480,
    )
    np.random.seed(0)
    env.reset()

    # Act
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(0)

    # Assert
    assert terminated
    assert env.primitive_steps < 430
