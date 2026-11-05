"""Tests for the meta agents without learning."""

from types import SimpleNamespace

import numpy as np
import pytest

from HUGIN_gym.evaluation.baselines import RuleChooser, make_baseline

SWEEP, PLUME, BORDER = 0, 1, 2
N_CORNERS = 4


def make_env(plume_share: float = 1.0, border_share: float = 1.0, plume_cells: int = 10) -> SimpleNamespace:
    """The parts of the HierarchicalEnv that the baselines read."""
    base = SimpleNamespace(
        maxN_over_thresh=N_CORNERS + plume_cells,
        state={
            key: np.zeros((4, 4), dtype=bool)
            for key in ("visited_map", "c_over_threshold_map", "c_around_threshold_map")
        },
    )
    env = SimpleNamespace(
        current_obs={},
        accuracy_goals=[0.9, 0.9, 0.9],
        primitive_steps=0,
        plume_share=plume_share,
        border_share=border_share,
        action_space=SimpleNamespace(n=3),
    )
    env.base_env = SimpleNamespace(
        unwrapped=base, N_corners=N_CORNERS, goal_shares=lambda obs: (env.plume_share, env.border_share)
    )
    return env


def make_progress(env: SimpleNamespace, cell: int) -> None:
    env.primitive_steps += 3
    env.base_env.unwrapped.state["visited_map"].flat[cell] = True


def test_rule_uses_the_first_option_while_no_plume_is_found() -> None:
    # Arrange: the coverages show 1.0 before a plume is found
    env = make_env(plume_cells=0)

    # Act and Assert
    assert RuleChooser(env)() == SWEEP


def test_rule_uses_the_first_option_when_plume_and_border_are_finished() -> None:
    # Arrange
    env = make_env(plume_share=0.95, border_share=0.92)

    # Act and Assert
    assert RuleChooser(env)() == SWEEP


@pytest.mark.parametrize(
    ("plume_share", "border_share", "expected"), [(0.2, 0.5, PLUME), (0.5, 0.2, BORDER), (0.95, 0.5, BORDER)]
)
def test_rule_picks_the_gas_sub_agent_with_the_lower_share(
    plume_share: float, border_share: float, expected: int
) -> None:
    # Arrange
    env = make_env(plume_share, border_share)

    # Act and Assert
    assert RuleChooser(env)() == expected


def test_rule_tries_the_other_gas_sub_agent_and_then_the_first_option_when_stuck() -> None:
    # Arrange
    env = make_env(plume_share=0.2, border_share=0.5)
    rule = RuleChooser(env)

    # Act: no new cells between the decisions
    first = rule()
    env.primitive_steps += 3
    second = rule()
    env.primitive_steps += 3
    third = rule()

    # Assert
    assert (first, second, third) == (PLUME, BORDER, SWEEP)


def test_rule_goes_back_to_the_lower_share_after_progress() -> None:
    # Arrange
    env = make_env(plume_share=0.2, border_share=0.5)
    rule = RuleChooser(env)
    rule()
    env.primitive_steps += 3
    rule()  # stuck: the other gas sub-agent

    # Act
    make_progress(env, cell=0)
    choice = rule()

    # Assert
    assert choice == PLUME


def test_rule_forgets_being_stuck_in_a_new_episode() -> None:
    # Arrange
    env = make_env(plume_share=0.2, border_share=0.5)
    rule = RuleChooser(env)
    rule()
    env.primitive_steps += 3
    rule()

    # Act: same counts as before, but the survey starts again
    env.primitive_steps = 0
    choice = rule()

    # Assert
    assert choice == PLUME


@pytest.mark.parametrize("name", ["lawnmower", "space"])
def test_lawnmower_and_space_always_use_the_first_option(name: str) -> None:
    # Arrange
    model = make_baseline(name, make_env(plume_share=0.1), seed=0)

    # Act and Assert
    assert model.predict({})[0] == 0


def test_random_is_uniform_and_leaves_the_global_generator_alone() -> None:
    # Arrange: the env draws its episodes from np.random, so the baseline must not use it
    model = make_baseline("random", make_env(), seed=0)
    np.random.seed(0)
    expected_next = np.random.rand()
    np.random.seed(0)

    # Act
    choices = [model.predict({})[0] for _ in range(3000)]

    # Assert
    assert np.random.rand() == expected_next
    np.testing.assert_allclose(np.bincount(choices) / len(choices), 1 / 3, atol=0.03)


def test_unknown_baseline_raises() -> None:
    # Act and Assert
    with pytest.raises(ValueError, match="Unknown baseline"):
        make_baseline("oracle", make_env(), seed=0)
