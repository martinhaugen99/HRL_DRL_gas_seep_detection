"""Tests that the META observations of GPWrapper come from the GP belief, never from the true plume."""

import numpy as np

from HUGIN_gym.envs.HUGIN_env import HUGIN
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper

PLUME_KEYS = [
    "local_GT",
    "local_GT_around",
    "GT_c_over_threshold_maps_downsampled",
    "GT_c_around_threshold_maps_downsampled",
    "plume_coverage",
    "border_coverage",
]
UP = 1  # outside the 2D map, so the base env rejects the move with -4


def make_meta_env() -> tuple[HUGIN, GPWrapper]:
    """META env as in the HRL training, with the AUV starting in the centre of the plume."""
    base = HUGIN()
    # fixed start at (0, 0), and the plume set below is kept at reset
    base.train = False
    base.GP_ON = True
    base.agent_type = "META"  # type: ignore[attr-defined]  # set by the scripts, not in HUGIN.__init__
    base.gaussian_centers = np.array([[0.0, 0.0, 0.0]])
    return base, GPWrapper(base, HRL=True)


def assert_shows_empty_belief(base: HUGIN, gp: GPWrapper, obs: dict) -> None:
    """The belief before the first GP update: no plume, only the corner cells."""
    empty_belief = base.downsample_map(gp.corner_mask).flatten()
    assert obs["local_GT"].max() <= 0.5  # 1.0 is a plume cell, 0.5 no plume
    assert obs["local_GT_around"].max() <= 0.5
    np.testing.assert_array_equal(
        obs["GT_c_over_threshold_maps_downsampled"], empty_belief
    )
    np.testing.assert_array_equal(
        obs["GT_c_around_threshold_maps_downsampled"], empty_belief
    )
    # the 4 corner cells, all visited
    assert obs["plume_coverage"][0] == obs["border_coverage"][0] == 1.0


def test_reset_observation_shows_belief_not_true_plume() -> None:
    # Arrange
    base, gp = make_meta_env()

    # Act
    obs, _ = gp.reset()

    # Assert: the true plume covers the start, but the observation does not show it
    ix, iy, iz = base._pos_to_idx(base.state["x"], base.state["y"], base.state["z"])
    assert (
        base.get_local_patch(gp.original_GT_c_over, ix, iy, iz, base.patch_radius).max()
        == 1.0
    )
    assert_shows_empty_belief(base, gp, obs)


def test_boundary_step_after_reset_shows_belief_not_true_plume() -> None:
    # Arrange
    base, gp = make_meta_env()
    gp.reset()

    # Act
    obs, reward, *_ = gp.step(UP)

    # Assert
    assert reward == -4.0
    assert_shows_empty_belief(base, gp, obs)


def test_boundary_step_repeats_the_latest_belief() -> None:
    # Arrange
    _, gp = make_meta_env()
    gp.reset()
    obs_move, *_ = gp.step(0)  # forward: the GP is updated

    # Act
    obs_boundary, reward, *_ = gp.step(UP)

    # Assert
    assert reward == -4.0
    for key in PLUME_KEYS:
        np.testing.assert_array_equal(obs_boundary[key], obs_move[key], err_msg=key)
