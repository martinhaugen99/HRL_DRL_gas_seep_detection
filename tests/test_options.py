"""Tests for Option.value (the sub-agent's critic estimate, from its own observation keys only) and the scripted options."""

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
    GasFillOption,
    HandoverOption,
    Option,
    SweepOption,
    survey_options,
)
from HUGIN_gym.envs.HRL_env import HierarchicalEnv
from HUGIN_gym.envs.HUGIN_env import HUGIN
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
from HUGIN_gym.evaluation.baselines import make_baseline

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


def make_gas_fill(goal_mode: str = "visit") -> tuple[HUGIN, SimpleNamespace, GasFillOption]:
    """A gas option on the real base env, with an empty belief and a GP that knows every cell moderately well."""
    base = HUGIN()
    base.HRL = True  # the base env gives no reward except the boundary penalty
    base.reset()
    base.state["x"], base.state["y"], base.state["theta"] = -10, -10, 0.0
    corners = np.zeros((41, 41, 1), dtype=bool)
    corners[[0, 0, 40, 40], [0, 40, 0, 40]] = True
    base.GT_c_over_threshold_maps = corners.copy()
    base.GT_c_around_threshold_maps = corners.copy()
    gp = SimpleNamespace(GP_std=np.full((41, 41), 0.5), goal_mode=goal_mode, known_std=0.1, corner_mask=corners)
    return base, gp, GasFillOption(base, gp)


def steps_until_near(base: HUGIN, option: GasFillOption, cell: tuple[int, int], max_steps: int = 40) -> int | None:
    """Follow the option until the AUV is within 1 cell of cell (x, y). None if it does not get there."""
    for step in range(max_steps):
        _, reward, *_ = base.step(option.act({}))
        assert reward != BOUNDARY_PENALTY
        if abs(base.state["x"] - cell[0]) <= 1 and abs(base.state["y"] - cell[1]) <= 1:
            return step + 1
    return None


def test_gas_fill_goes_to_the_nearest_unfinished_cell() -> None:
    # Arrange: a border cell near the AUV and a plume cell far away
    base, _, option = make_gas_fill()
    base.GT_c_around_threshold_maps[0 + 20, -10 + 20] = True
    base.GT_c_over_threshold_maps[15 + 20, 15 + 20] = True

    # Act
    steps = steps_until_near(base, option, (0, -10))

    # Assert: 10 cells away, and the forward action moves 2
    assert steps == 5


def test_gas_fill_with_the_visit_goal_skips_visited_cells() -> None:
    # Arrange: the nearest gas cell is visited
    base, _, option = make_gas_fill(goal_mode="visit")
    base.GT_c_over_threshold_maps[-8 + 20, -10 + 20] = True
    base.state["visited_map"][-8 + 20, -10 + 20, 0, 0] = True
    base.GT_c_over_threshold_maps[10 + 20, -10 + 20] = True

    # Act
    steps = steps_until_near(base, option, (10, -10))

    # Assert
    assert steps == 10


def test_gas_fill_with_the_estimate_goal_skips_known_cells_even_if_not_visited() -> None:
    # Arrange: the nearest gas cell is well known, but the AUV has not been there
    base, gp, option = make_gas_fill(goal_mode="estimate")
    base.GT_c_over_threshold_maps[-8 + 20, -10 + 20] = True
    gp.GP_std[-8 + 20, -10 + 20] = 0.05
    base.GT_c_over_threshold_maps[10 + 20, -10 + 20] = True

    # Act
    steps = steps_until_near(base, option, (10, -10))

    # Assert
    assert steps == 10


def test_gas_fill_with_the_estimate_goal_returns_to_visited_cells_that_are_not_known() -> None:
    # Arrange
    base, _, option = make_gas_fill(goal_mode="estimate")
    base.GT_c_over_threshold_maps[0 + 20, -10 + 20] = True
    base.state["visited_map"][0 + 20, -10 + 20, 0, 0] = True

    # Act and Assert
    assert steps_until_near(base, option, (0, -10)) == 5


def test_gas_fill_with_only_the_corner_cells_goes_to_the_least_known_cells() -> None:
    # Arrange: no gas found, and the corners do not count as gas
    base, gp, option = make_gas_fill()
    gp.GP_std[0 + 18:0 + 23, 0 + 18:0 + 23] = 1.0  # around the middle of the map, more than 1 % of it

    # Act
    steps = steps_until_near(base, option, (0, 0))

    # Assert
    assert steps is not None


@pytest.mark.parametrize(
    ("goal_mode", "spacing", "goal_mean_std"),
    [("visit", 4, 0.1), ("estimate", 6, 0.26)],
)
def test_rule_with_the_scripted_options_reaches_the_goal_with_several_plumes(
    goal_mode: str, spacing: int, goal_mean_std: float
) -> None:
    # Arrange: the real env of the HRL evaluation
    base = HUGIN()
    base.train = True
    base.use_c_map = True
    base.GP_ON = True
    base.HRL = True
    base.agent_type = "META"  # type: ignore[attr-defined]  # set by the scripts, not in HUGIN.__init__
    base.spawn_at_survey_start = True
    base.vary_plume_size = True
    base.multiple_gaussians = [2, 2]
    gp_env = GPWrapper(base, HRL=True, goal_mode=goal_mode, goal_mean_std=goal_mean_std)
    env = HierarchicalEnv(
        gp_env,
        [SweepOption(base, gp_env, spacing), GasFillOption(base, gp_env), GasFillOption(base, gp_env)],
        gamma=0.998,
        MAX_OPT_LENGTH=3,
        ACCURACY_GOALS=[0.9, 0.9, 0.9],
        GP=True,
        max_primitive_steps=480,
    )
    model = make_baseline("rule", env, seed=0)
    np.random.seed(1)
    obs, _ = env.reset()

    # Act
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    # Assert
    assert terminated
    assert env.primitive_steps < 480


class RecordingAgent:
    """Stands in for a trained plume or border agent."""

    name = "trained agent"

    def __init__(self) -> None:
        self.calls = 0

    def act(self, obs: dict) -> int:
        self.calls += 1
        return LEFT

    def value(self, obs: dict) -> float:
        return 12.0


def make_handover(goal_mode: str = "visit") -> tuple[HUGIN, SimpleNamespace, RecordingAgent, HandoverOption]:
    base, gp, _ = make_gas_fill(goal_mode)
    agent = RecordingAgent()
    return base, gp, agent, HandoverOption(base, gp, agent, "GT_c_around_threshold_maps")


def test_handover_goes_to_the_target_while_the_agent_cannot_see_it() -> None:
    # Arrange: a border cell 10 cells ahead, outside the 7 x 7 patch
    base, _, agent, option = make_handover()
    base.GT_c_around_threshold_maps[0 + 20, -10 + 20] = True

    # Act
    action = option.act({})

    # Assert
    assert action == FORWARD
    assert agent.calls == 0


def test_handover_lets_the_trained_agent_decide_when_it_sees_the_target() -> None:
    # Arrange: a border cell 3 cells ahead, inside the patch
    base, _, agent, option = make_handover()
    base.GT_c_around_threshold_maps[-7 + 20, -10 + 20] = True

    # Act
    action = option.act({})

    # Assert
    assert action == LEFT
    assert agent.calls == 1


def test_handover_brings_the_auv_into_view_and_then_hands_over() -> None:
    # Arrange
    base, _, agent, option = make_handover()
    base.GT_c_around_threshold_maps[0 + 20, -10 + 20] = True

    # Act
    for _ in range(4):
        base.step(FORWARD if agent.calls == 0 else option.act({}))
        option.act({})
        if agent.calls:
            break

    # Assert: 10 cells away: after 2 forward moves of 2 cells it is 6 away, after 4 moves it is within 3 cells
    assert agent.calls == 1


def test_handover_counts_only_the_map_of_the_agent() -> None:
    # Arrange: plume cells near the AUV, and the border agent's own cell far away
    base, _, agent, option = make_handover()
    base.GT_c_over_threshold_maps[-8 + 20, -10 + 20] = True
    base.GT_c_around_threshold_maps[10 + 20, -10 + 20] = True

    # Act
    option.act({})

    # Assert
    assert agent.calls == 0


def test_handover_with_the_estimate_goal_targets_cells_that_are_not_known() -> None:
    # Arrange: a known border cell nearby, and an unknown one far away
    base, gp, agent, option = make_handover("estimate")
    base.GT_c_around_threshold_maps[-8 + 20, -10 + 20] = True
    gp.GP_std[-8 + 20, -10 + 20] = 0.05
    base.GT_c_around_threshold_maps[10 + 20, -10 + 20] = True

    # Act
    option.act({})

    # Assert: it does not hand over for the known cell
    assert agent.calls == 0


def test_handover_with_no_target_left_lets_the_trained_agent_decide() -> None:
    # Arrange
    _, _, agent, option = make_handover()

    # Act
    option.act({})

    # Assert
    assert agent.calls == 1


def test_handover_keeps_the_name_and_the_value_of_the_trained_agent() -> None:
    # Arrange
    _, _, _, option = make_handover()

    # Act and Assert: minimal_values reads the critic of the sub-agents
    assert option.name == "trained agent"
    assert option.value({}) == 12.0


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, (RecordingAgent, RecordingAgent, RecordingAgent)),
        ({"spacing": 6}, (SweepOption, RecordingAgent, RecordingAgent)),
        ({"handover": True}, (RecordingAgent, HandoverOption, HandoverOption)),
        ({"spacing": 6, "handover": True}, (SweepOption, HandoverOption, HandoverOption)),
        ({"spacing": 6, "scripted_gas": True}, (SweepOption, GasFillOption, GasFillOption)),
    ],
)
def test_survey_options_builds_the_options_of_a_survey(settings: dict, expected: tuple) -> None:
    # Arrange
    base, gp, _ = make_gas_fill()
    trained = [RecordingAgent(), RecordingAgent(), RecordingAgent()]

    # Act
    options = survey_options(trained, base, gp, **settings)

    # Assert
    assert tuple(type(option) for option in options) == expected


def test_survey_options_gives_each_gas_agent_its_own_map() -> None:
    # Arrange
    base, gp, _ = make_gas_fill()

    # Act
    _, plume, border = survey_options([None, RecordingAgent(), RecordingAgent()], base, gp, spacing=6, handover=True)

    # Assert
    assert plume.belief_map == "GT_c_over_threshold_maps"
    assert border.belief_map == "GT_c_around_threshold_maps"
