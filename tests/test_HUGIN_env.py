"""Tests for the start position and plume size options of the HUGIN env."""

import numpy as np

from HUGIN_gym.envs.core.visualisation.renderer import concentration_bands
from HUGIN_gym.envs.HUGIN_env import HUGIN


def make_env(**flags: bool) -> HUGIN:
    """Env with a random start and plume as in training, without a sub-agent reward as in the HRL training."""
    env = HUGIN()
    env.train = True
    env.HRL = True
    env.agent_type = "META"  # type: ignore[attr-defined]  # set by the scripts, not in HUGIN.__init__
    for name, value in flags.items():
        setattr(env, name, value)
    return env


def test_survey_start_is_a_corner_heading_along_an_edge_into_the_map() -> None:
    # Arrange
    env = make_env(spawn_at_survey_start=True)
    corner = (env.domain_limit["x"] - 1, env.domain_limit["y"] - 1)
    np.random.seed(0)
    starts = set()

    for _ in range(50):
        # Act
        env.reset()
        x, y, theta = env.state["x"], env.state["y"], env.state["theta"]
        _, reward, *_ = env.step(0)  # forward

        # Assert: the first move forward stays in the map and on one of the two edges
        assert (abs(x), abs(y)) == corner
        assert reward != -4.0
        assert (env.state["x"], env.state["y"]) != (x, y)
        assert abs(env.state["x"]) == corner[0] or abs(env.state["y"]) == corner[1]
        starts.add((x, y, theta))

    assert len(starts) == 8  # 4 corners with 2 headings each


def test_vary_plume_size_draws_a_new_sigma_from_the_range() -> None:
    # Arrange
    env = make_env(vary_plume_size=True)
    low, high = env.gaussian_sigma_range
    np.random.seed(0)

    # Act
    sigmas = []
    for _ in range(20):
        env.reset()
        sigmas.extend(np.atleast_1d(env.gaussian_sigma))

    # Assert
    assert all(low <= sigma <= high for sigma in sigmas)
    assert len(set(sigmas)) == len(sigmas)


def test_cloud_bands_in_the_view_are_the_plume_and_border_of_the_coverage() -> None:
    # Arrange
    env = make_env(vary_plume_size=True)
    np.random.seed(0)

    for _ in range(5):
        env.reset()

        # Act
        bands = concentration_bands(env.conc, env.c_threshold, env.c_around_width)

        # Assert
        assert bands["plume"].any() and bands["border"].any()
        np.testing.assert_array_equal(
            bands["plume"] | bands["plume and border"], env.GT_c_over_threshold_maps
        )
        np.testing.assert_array_equal(
            bands["plume and border"] | bands["border"], env.GT_c_around_threshold_maps
        )
