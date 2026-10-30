"""Tests for the option loop of HierarchicalEnv: discounting inside an option, option lengths, rewards, meta states."""

import gymnasium as gym
import numpy as np
import pytest

from HUGIN_gym.envs.HRL_env import HierarchicalEnv

GAMMA = 0.9
SPACE_STEP_REWARD = 1 / 6 - 1  # space option: one new cell per step, minus the time penalty


class StubRenderer:
    def __init__(self) -> None:
        self.labels: list[tuple[str, int]] = []

    def show_label(self, lines: list[tuple[str, int]]) -> None:
        assert len(lines) == 1
        self.labels.append(lines[0])


class StubBaseEnv:
    """Stand-in for the GP-wrapped HUGIN env. The AUV visits one new cell per step if progress is set, otherwise
    none, and the episode ends after end_step steps. Plume and border are fully covered."""

    N_corners = 4  # as GPWrapper

    def __init__(self, end_step: int | None = None, progress: bool = False) -> None:
        empty_map = np.zeros((10, 10, 1, 1), dtype=bool)
        self.state = {
            key: empty_map.copy()
            for key in ("visited_map", "c_over_threshold_map", "c_around_threshold_map")
        }
        self.old_visited_sum = self.old_c_over_sum = self.old_c_around_sum = 0
        coverage_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {key: coverage_space for key in ("space_coverage", "plume_coverage", "border_coverage")}
        )
        self.end_step = end_step
        self.progress = progress
        self.steps = 0
        self.maxN_over_thresh = 4.0  # only the corner cells: no plume found
        self.MEAN_uncertainty = 1.0
        self.renderer = StubRenderer()
        self.marker_colors: list[int] = []

    def step_sim(self, color: int) -> None:
        self.marker_colors.append(color)

    @property
    def unwrapped(self) -> "StubBaseEnv":
        return self

    def reset(self, **kwargs: object) -> tuple[dict, dict]:
        self.steps = 0
        self.state["visited_map"][:] = False
        return self._obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        self.steps += 1
        self.old_visited_sum = np.sum(self.state["visited_map"])
        if self.progress:
            self.state["visited_map"].flat[self.steps - 1] = True
        terminated = self.end_step is not None and self.steps >= self.end_step
        return self._obs(), 0.0, terminated, False, {}

    def _obs(self) -> dict:
        covered = np.array([1.0], dtype=np.float32)  # plume and border covered
        space = np.array([self.state["visited_map"].mean()], dtype=np.float32)
        return {"space_coverage": space, "plume_coverage": covered, "border_coverage": covered}


class StubOption:
    def __init__(self, value_estimate: float = 0.0, name: str = "option") -> None:
        self.value_estimate = value_estimate
        self.name = name

    def act(self, obs: dict) -> int:
        return 0

    def value(self, obs: dict) -> float:
        return self.value_estimate


def make_env(
    max_opt_length: int = 3,
    end_step: int | None = None,
    progress: bool = False,
    options: list[StubOption] | None = None,
    **settings: object,
) -> HierarchicalEnv:
    env = HierarchicalEnv(
        StubBaseEnv(end_step, progress),
        options or [StubOption()],
        gamma=GAMMA,
        MAX_OPT_LENGTH=max_opt_length,
        ACCURACY_GOALS=[0.9, 0.9, 0.9],
        GP=True,
        **settings,
    )
    env.reset()
    return env


def discounted_sum(step_reward: float, steps: int) -> float:
    return step_reward * sum(GAMMA**i for i in range(steps))


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


@pytest.mark.parametrize(
    ("meta_action", "option", "length"),
    [(0, 0, 2), (1, 0, 5), (2, 1, 2), (3, 1, 5)],
)
def test_meta_action_picks_option_and_length(meta_action: int, option: int, length: int) -> None:
    # Arrange
    env = make_env(progress=True, options=[StubOption(), StubOption()], option_lengths=[2, 5], smdp_discount=True)

    # Act
    _, _, _, _, info = env.step(meta_action)

    # Assert
    assert env.action_space.n == 4
    assert (info["option"], info["option_length"]) == (option, length)
    names = HierarchicalEnv.action_names(["space", "plume"], [2, 5])
    assert names[meta_action] == f"{['space', 'plume'][option]}_{length}"


def test_smdp_decision_reports_its_discount_and_sums_its_rewards() -> None:
    # Arrange
    env = make_env(progress=True, option_lengths=[2, 5], smdp_discount=True, decision_cost=0.1)

    # Act: the 5-step variant of the space option
    _, reward, _, _, info = env.step(1)

    # Assert
    assert info["discount"] == pytest.approx(GAMMA**5)
    assert reward == pytest.approx(discounted_sum(SPACE_STEP_REWARD, 5) - 0.1)


def test_mean_option_reward_divides_the_step_rewards_by_the_steps() -> None:
    # Arrange
    env = make_env(progress=True, option_lengths=[2, 5], mean_option_reward=True, decision_cost=0.1)

    # Act
    _, reward, _, _, _ = env.step(1)

    # Assert
    assert reward == pytest.approx(discounted_sum(SPACE_STEP_REWARD, 5) / 5 - 0.1)


def test_smdp_option_may_end_early_without_progress() -> None:
    # Arrange
    env = make_env(option_lengths=[2, 5], smdp_discount=True)

    # Act: 3 steps without progress end the 5-step option
    _, reward, terminated, truncated, info = env.step(1)

    # Assert
    assert info["option_length"] == 3
    assert info["discount"] == pytest.approx(GAMMA**3)
    assert reward == pytest.approx(-(1 + GAMMA + GAMMA**2) - 2.0)
    assert not (terminated or truncated)


def test_survey_budget_truncates_the_episode() -> None:
    # Arrange
    env = make_env(max_opt_length=3, progress=True, max_primitive_steps=4)

    # Act
    _, _, _, first_truncated, _ = env.step(0)
    _, _, terminated, truncated, info = env.step(0)

    # Assert: the second option is cut after 1 step
    assert not first_truncated
    assert truncated and not terminated
    assert info["option_length"] == 1
    assert info["survey_steps"] == 4
    assert info["survey_fraction"] == pytest.approx(3 / 4)


@pytest.mark.parametrize(
    "settings",
    [
        {"option_lengths": [2, 5]},  # needs smdp_discount or mean_option_reward
        {"option_lengths": [2, 5], "smdp_discount": True, "mean_option_reward": True},
        {"meta_state": "minimal"},  # needs max_primitive_steps
        {"meta_state": "small", "max_primitive_steps": 12},
    ],
)
def test_invalid_settings_raise(settings: dict) -> None:
    # Act and Assert
    with pytest.raises(ValueError):
        make_env(**settings)


def test_full_meta_state_is_the_base_observation() -> None:
    # Arrange
    env = make_env()

    # Act
    obs, _ = env.reset()

    # Assert
    assert env.observation_space is env.base_env.observation_space
    assert set(obs) == {"space_coverage", "plume_coverage", "border_coverage"}


def test_minimal_meta_state() -> None:
    # Arrange
    env = make_env(progress=True, meta_state="minimal", max_primitive_steps=12)
    reset_obs, _ = env.reset()

    # Act: a plume is found and the GP gets more certain during the first option
    env.base_env.maxN_over_thresh = 10.0
    env.base_env.MEAN_uncertainty = 0.4
    obs, _, _, _, _ = env.step(0)

    # Assert
    assert reset_obs["time_used"][0] == 0.0
    assert reset_obs["plume_found"][0] == 0.0
    assert obs["time_used"][0] == pytest.approx(3 / 12)
    assert obs["plume_found"][0] == 1.0
    assert obs["gp_uncertainty"][0] == pytest.approx(0.4)
    assert obs["space_coverage"][0] == pytest.approx(3 / 100)
    assert env.observation_space.contains(obs)


def test_visualised_decision_labels_the_sub_agent_and_colors_its_steps() -> None:
    # Arrange
    env = make_env(progress=True, options=[StubOption(name="spatial exploration"), StubOption(name="gas plume exploration")])
    env.visualise_env = True
    env.option_colors = [0x00FF00, 0xFF3399]

    # Act
    env.step(1)
    env.step(0)

    # Assert: one label per decision, one marker per AUV step in the color of the option
    assert env.base_env.renderer.labels == [
        ("Sub-agent: gas plume exploration", 0xFF3399),
        ("Sub-agent: spatial exploration", 0x00FF00),
    ]
    assert env.base_env.marker_colors == [0xFF3399] * 3 + [0x00FF00] * 3


def test_visualised_label_shows_the_option_length_if_there_are_several() -> None:
    # Arrange
    env = make_env(progress=True, options=[StubOption(name="spatial exploration")], option_lengths=[2, 5], smdp_discount=True)
    env.visualise_env = True

    # Act
    env.step(1)

    # Assert
    assert env.base_env.renderer.labels == [("Sub-agent: spatial exploration (5 steps)", 0x00FF00)]


def test_not_visualised_by_default() -> None:
    # Arrange
    env = make_env(progress=True)

    # Act
    env.step(0)

    # Assert
    assert env.base_env.renderer.labels == []
    assert env.base_env.marker_colors == []


def test_minimal_values_meta_state_adds_the_sub_agent_values() -> None:
    # Arrange
    env = make_env(
        options=[StubOption(10.0), StubOption(-3.0)], meta_state="minimal_values", max_primitive_steps=12
    )

    # Act
    obs, _ = env.reset()

    # Assert: symlog of the values
    np.testing.assert_allclose(obs["subagent_values"], [np.log1p(10.0), -np.log1p(3.0)], rtol=1e-6)
    assert env.observation_space.contains(obs)
