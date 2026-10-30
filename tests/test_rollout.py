"""Tests for the HRL rollout in HUGIN_gym.evaluation.rollout."""

import numpy as np
import pytest

from HUGIN_gym.evaluation.rollout import run_episode_hrl

OPTION_LENGTH = 3


class StubRenderer:
    """Records the Meshcat calls of the rollout in a shared event list."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.vis = self

    def delete(self) -> None:
        self.events.append("delete")

    def reset(self) -> None:
        self.events.append("reset")


class StubBaseEnv:
    """Stand-in for the GP-wrapped HUGIN env, with the attributes the rollout reads and sets."""

    train = True
    option_length = 0

    def __init__(self) -> None:
        self.events: list[str] = []
        self.renderer = StubRenderer(self.events)

    @property
    def unwrapped(self) -> "StubBaseEnv":
        return self

    def render(self) -> None:
        self.events.append("render")

    def step_sim(self) -> None:
        self.events.append("step_sim")


class StubHierarchicalEnv:
    """Stand-in for HierarchicalEnv: every decision runs OPTION_LENGTH steps, and the survey budget ends the
    episode after n_decisions."""

    visualise_env = False

    def __init__(self, n_decisions: int) -> None:
        self.base_env = StubBaseEnv()
        self.turn_counter = 0
        self.n_decisions = n_decisions
        self.decisions = 0

    def reset(self, **kwargs: object) -> tuple[dict, dict]:
        self.decisions = 0
        return self._obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        self.decisions += 1
        truncated = self.decisions >= self.n_decisions
        return (
            self._obs(),
            -1.0,
            False,
            truncated,
            {"option": action, "option_length": OPTION_LENGTH},
        )

    @staticmethod
    def _obs() -> dict:
        return {
            key: np.array([0.5], dtype=np.float32)
            for key in ("space_coverage", "plume_coverage", "border_coverage")
        }


class StubModel:
    """Meta agent that plays the given actions in order and records the deterministic flag of each call."""

    def __init__(self, actions: list[int]) -> None:
        self.actions = actions
        self.deterministic_flags: list[bool] = []

    def predict(
        self, obs: dict, deterministic: bool = False
    ) -> tuple[np.ndarray, None]:
        self.deterministic_flags.append(deterministic)
        return np.array(self.actions[len(self.deterministic_flags) - 1]), None


@pytest.mark.parametrize("deterministic", [True, False])
def test_run_episode_hrl_passes_deterministic_to_the_model(deterministic: bool) -> None:
    # Arrange
    env = StubHierarchicalEnv(n_decisions=3)
    model = StubModel(actions=[0, 1, 2])

    # Act
    run_episode_hrl(env, model, max_eps_len=10, deterministic=deterministic)

    # Assert
    assert model.deterministic_flags == [deterministic] * 3


def test_run_episode_hrl_deterministic_by_default() -> None:
    # Arrange
    env = StubHierarchicalEnv(n_decisions=2)
    model = StubModel(actions=[0, 0])

    # Act
    run_episode_hrl(env, model, max_eps_len=10)

    # Assert
    assert model.deterministic_flags == [True, True]


def test_run_episode_hrl_returns_the_meta_actions_in_order() -> None:
    # Arrange
    env = StubHierarchicalEnv(n_decisions=4)
    model = StubModel(actions=[0, 2, 2, 1])

    # Act
    result = run_episode_hrl(env, model, max_eps_len=10)

    # Assert
    assert result["actions"] == [0, 2, 2, 1]
    assert result["options_chosen"] == 4
    assert result["base_env_steps"] == 4 * OPTION_LENGTH


@pytest.mark.parametrize(
    ("visualise_env", "markers_per_decision"),
    [(False, 1), (True, 0)],  # with visualise_env, the env draws every AUV step itself
)
def test_run_episode_hrl_draws_markers_only_without_visualise_env(
    visualise_env: bool, markers_per_decision: int
) -> None:
    # Arrange
    env = StubHierarchicalEnv(n_decisions=2)
    env.visualise_env = visualise_env

    # Act
    run_episode_hrl(env, StubModel(actions=[0, 1]), max_eps_len=10, VISUALISE=True)

    # Assert
    assert env.base_env.events.count("step_sim") == 2 * markers_per_decision


def test_run_episode_hrl_pauses_before_clearing_the_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    env = StubHierarchicalEnv(n_decisions=2)
    env.visualise_env = True
    monkeypatch.setattr(
        "builtins.input", lambda prompt: env.base_env.events.append("pause")
    )

    # Act
    run_episode_hrl(
        env,
        StubModel(actions=[0, 1]),
        max_eps_len=10,
        VISUALISE=True,
        pause_at_end=True,
    )

    # Assert
    assert env.base_env.events == ["delete", "render", "pause", "reset"]
