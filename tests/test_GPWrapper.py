"""Tests that the META observations of GPWrapper come from the GP belief, never from the true plume, and for its fast GP and end goals."""

import numpy as np
import pytest

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


FORWARD, LEFT, RIGHT = 0, 3, 4
ACTIONS = [FORWARD, FORWARD, LEFT, FORWARD, UP, RIGHT, FORWARD, FORWARD, LEFT, LEFT, FORWARD, RIGHT]


def run_survey(train: bool, **gp_settings: object) -> tuple[HUGIN, GPWrapper, list[dict]]:
    """The same plume and the same moves for every call."""
    np.random.seed(0)  # without train, the plume is drawn when the env is created
    base = HUGIN()
    base.train = train
    base.GP_ON = True
    base.agent_type = "META"  # type: ignore[attr-defined]  # set by the scripts, not in HUGIN.__init__
    base.vary_plume_size = True
    gp = GPWrapper(base, HRL=True, **gp_settings)
    gp.reset()
    return base, gp, [gp.step(action)[0] for action in ACTIONS]


@pytest.mark.parametrize("train", [True, False])
def test_fast_gp_gives_the_same_map_and_observations_as_sklearn(train: bool) -> None:
    # Act
    fast_base, fast_gp, fast_obs = run_survey(train, fast_gp=True)
    sklearn_base, sklearn_gp, sklearn_obs = run_survey(train, fast_gp=False)

    # Assert
    assert fast_gp.incremental_gp.n == len(fast_gp.X)
    assert sklearn_gp.incremental_gp is None
    np.testing.assert_allclose(fast_gp.GP_pred, sklearn_gp.GP_pred, atol=1e-8)
    np.testing.assert_allclose(fast_gp.GP_std, sklearn_gp.GP_std, atol=1e-7)
    assert fast_base.MEAN_uncertainty == pytest.approx(sklearn_base.MEAN_uncertainty, abs=1e-7)
    for fast_step, sklearn_step in zip(fast_obs, sklearn_obs, strict=True):
        for key in PLUME_KEYS:
            np.testing.assert_array_equal(fast_step[key], sklearn_step[key], err_msg=key)


def test_sklearn_takes_over_when_measurements_are_dropped() -> None:
    # Act: more measurements than max_points
    _, fast_gp, _ = run_survey(True, fast_gp=True, max_points=3)
    _, sklearn_gp, _ = run_survey(True, fast_gp=False, max_points=3)

    # Assert
    assert not fast_gp.incremental_gp_valid
    np.testing.assert_allclose(fast_gp.GP_pred, sklearn_gp.GP_pred, atol=1e-8)


def test_reset_starts_the_fast_gp_again() -> None:
    # Arrange
    _, gp, _ = run_survey(True)

    # Act
    gp.reset()

    # Assert
    assert gp.incremental_gp.n == 0
    assert gp.GP_std.min() == 1.0


def test_only_the_fixed_rbf_kernel_uses_the_fast_gp() -> None:
    # Arrange
    base, _ = make_meta_env()

    # Act
    matern = GPWrapper(base, kernels_config={"Matern": {"type": "Matern", "length_scale": 3.5, "nu": 1.5}})
    optimised = GPWrapper(base, gp_optimise=True)
    plain = GPWrapper(base)

    # Assert
    assert matern.incremental_gp is None
    assert optimised.incremental_gp is None
    assert plain.incremental_gp is not None


def make_goal_env(**goal_settings: object) -> tuple[HUGIN, GPWrapper, dict]:
    """A found plume of 4 cells and a border of 4 cells that the GP knows well, but that the AUV has not visited."""
    base, _ = make_meta_env()
    gp = GPWrapper(base, HRL=True, **goal_settings)
    gp.reset()
    over = np.zeros((41, 41, 1), dtype=bool)
    over[10:12, 10:12] = True
    around = np.zeros((41, 41, 1), dtype=bool)
    around[12:14, 10:12] = True
    base.GT_c_over_threshold_maps = over | gp.corner_mask
    base.GT_c_around_threshold_maps = around | gp.corner_mask
    base.maxN_over_thresh = float(np.sum(base.GT_c_over_threshold_maps))
    base.maxN_around_thresh = float(np.sum(base.GT_c_around_threshold_maps))
    gp.GP_std = np.full((41, 41), 0.5)
    gp.GP_std[10:14, 10:12] = 0.05
    base.MEAN_uncertainty = 0.2
    not_visited = {"plume_coverage": np.array([0.1]), "border_coverage": np.array([0.1])}
    return base, gp, not_visited


def test_visit_goal_needs_the_cells_to_be_visited() -> None:
    # Arrange
    _, gp, not_visited = make_goal_env(goal_mode="visit")

    # Act and Assert
    assert gp.goal_shares(not_visited) == (0.1, 0.1)
    assert gp._compute_meta_subgoal_reward(not_visited) == (0.0, False)


def test_estimate_goal_only_needs_the_cells_to_be_known() -> None:
    # Arrange
    _, gp, not_visited = make_goal_env(goal_mode="estimate", goal_mean_std=0.26)

    # Act and Assert: the known plume first gives the subgoal bonus, and the next check ends the episode
    assert gp.goal_shares(not_visited) == (1.0, 1.0)
    assert gp._compute_meta_subgoal_reward(not_visited) == (40.0, False)
    assert gp._compute_meta_subgoal_reward(not_visited) == (40.0, True)


def test_estimate_goal_waits_for_the_mean_std_of_the_map() -> None:
    # Arrange
    _, gp, not_visited = make_goal_env(goal_mode="estimate", goal_mean_std=0.1)

    # Act
    gp._compute_meta_subgoal_reward(not_visited)  # the subgoal bonus

    # Assert: 0.2 is above 0.1
    assert gp._compute_meta_subgoal_reward(not_visited) == (0.0, False)


def test_estimate_goal_counts_only_well_known_cells() -> None:
    # Arrange
    _, gp, not_visited = make_goal_env(goal_mode="estimate")
    gp.GP_std[10, 10] = 0.5  # 1 of the 4 plume cells is not known

    # Act
    plume_share, border_share = gp.goal_shares(not_visited)

    # Assert
    assert plume_share == pytest.approx(0.75)
    assert border_share == 1.0


def test_estimate_goal_with_no_gas_found_is_complete() -> None:
    # Arrange
    base, gp, not_visited = make_goal_env(goal_mode="estimate")
    base.GT_c_over_threshold_maps = gp.corner_mask.copy()
    base.GT_c_around_threshold_maps = gp.corner_mask.copy()

    # Act and Assert
    assert gp.goal_shares(not_visited) == (1.0, 1.0)


def test_estimate_goal_waits_for_a_large_unknown_area() -> None:
    # Arrange: the mean GP std is low enough, but part of the map has no measurements nearby
    _, gp, not_visited = make_goal_env(goal_mode="estimate", goal_mean_std=0.26, max_unknown_cells=10)
    gp.GP_std[30:35, 30:35] = 0.95  # 25 connected cells
    gp._compute_meta_subgoal_reward(not_visited)  # the subgoal bonus

    # Act and Assert
    assert gp.largest_unknown_area() == 25
    assert gp._compute_meta_subgoal_reward(not_visited) == (0.0, False)


def test_estimate_goal_ends_when_the_unknown_area_is_small_enough() -> None:
    # Arrange
    _, gp, not_visited = make_goal_env(goal_mode="estimate", goal_mean_std=0.26, max_unknown_cells=10)
    gp.GP_std[30:32, 30:32] = 0.95  # 4 connected cells
    gp._compute_meta_subgoal_reward(not_visited)

    # Act and Assert
    assert gp.largest_unknown_area() == 4
    assert gp._compute_meta_subgoal_reward(not_visited) == (40.0, True)


def test_unknown_areas_are_measured_one_by_one_not_added_up() -> None:
    # Arrange: two areas of 4 cells that do not touch
    _, gp, _ = make_goal_env(goal_mode="estimate")
    gp.GP_std[30:32, 30:32] = 0.95
    gp.GP_std[35:37, 35:37] = 0.95

    # Act and Assert
    assert gp.largest_unknown_area() == 4


def test_unknown_area_is_zero_when_the_whole_map_is_below_the_limit() -> None:
    # Arrange
    _, gp, _ = make_goal_env(goal_mode="estimate")

    # Act and Assert: 0.5 and 0.05 are below the limit of 0.8
    assert gp.largest_unknown_area() == 0


def test_visit_goal_ignores_unknown_areas() -> None:
    # Arrange: the plume is visited and the mean GP std is low, as today's goal needs
    base, gp, _ = make_goal_env(goal_mode="visit")
    visited = {"plume_coverage": np.array([1.0]), "border_coverage": np.array([1.0])}
    base.MEAN_uncertainty = 0.05
    gp.GP_std[30:35, 30:35] = 0.95
    gp._compute_meta_subgoal_reward(visited)

    # Act and Assert
    assert gp._compute_meta_subgoal_reward(visited) == (40.0, True)


def test_unknown_goal_mode_raises() -> None:
    # Act and Assert
    with pytest.raises(ValueError, match="goal_mode"):
        GPWrapper(HUGIN(), goal_mode="guess")


def test_map_score_is_the_mean_overlap_with_the_true_map() -> None:
    # Arrange: the estimate is exactly the true concentration
    base, gp = make_meta_env()
    gp.reset()
    empty_score = gp.map_score()

    # Act
    gp.GP_pred = base.conc[:, :, 0].astype(float)
    perfect_score = gp.map_score()

    # Assert
    assert empty_score == 0.0
    assert perfect_score > 0.99


def make_bonus_env() -> tuple[HUGIN, GPWrapper, dict]:
    """A plume patch of 25 cells that the GP knows well, with a low mean GP std, but too high to end the episode."""
    base, gp, not_visited = make_goal_env(goal_mode="estimate", goal_mean_std=0.01, bonus_mode="patch")
    base.GT_c_over_threshold_maps[:] = gp.corner_mask
    base.GT_c_around_threshold_maps[:] = gp.corner_mask
    base.GT_c_over_threshold_maps[10:15, 10:15] = True
    gp.GP_std[:] = 0.5
    gp.GP_std[10:15, 10:15] = 0.05
    return base, gp, not_visited


def test_patch_bonus_is_paid_once_for_a_finished_patch() -> None:
    # Arrange
    _, gp, obs = make_bonus_env()

    # Act and Assert: GP noise around the goal does not pay again
    assert gp._compute_meta_subgoal_reward(obs) == (40.0, False)
    assert gp._compute_meta_subgoal_reward(obs) == (0.0, False)
    gp.GP_std[10:15, 10] = 0.5  # the patch falls below the goal
    assert gp._compute_meta_subgoal_reward(obs) == (0.0, False)
    gp.GP_std[10:15, 10] = 0.05  # and reaches it again
    assert gp._compute_meta_subgoal_reward(obs) == (0.0, False)


def test_patch_bonus_is_paid_again_for_a_second_plume() -> None:
    # Arrange
    base, gp, obs = make_bonus_env()
    gp._compute_meta_subgoal_reward(obs)

    # Act: a second plume is found and finished
    base.GT_c_over_threshold_maps[25:30, 25:30] = True
    gp.GP_std[25:30, 25:30] = 0.05

    # Assert
    assert gp._compute_meta_subgoal_reward(obs) == (40.0, False)
    assert gp._compute_meta_subgoal_reward(obs) == (0.0, False)


def test_patch_bonus_waits_until_the_patch_reaches_the_goal() -> None:
    # Arrange: half of the patch is not known
    _, gp, obs = make_bonus_env()
    gp.GP_std[10:15, 10:12] = 0.5

    # Act and Assert
    assert gp._compute_meta_subgoal_reward(obs) == (0.0, False)


def test_patch_bonus_ignores_a_patch_too_small_to_be_a_plume() -> None:
    # Arrange: 4 cells, as when the GP has only just found gas
    base, gp, obs = make_bonus_env()
    base.GT_c_over_threshold_maps[:] = gp.corner_mask
    base.GT_c_over_threshold_maps[10:12, 10:12] = True

    # Act and Assert
    assert gp._compute_meta_subgoal_reward(obs) == (0.0, False)


def test_no_bonus_mode_still_ends_the_episode() -> None:
    # Arrange
    _, gp, obs = make_goal_env(goal_mode="estimate", goal_mean_std=0.26, bonus_mode="none")

    # Act and Assert: the end check comes at once, with no bonus before it
    assert gp._compute_meta_subgoal_reward(obs) == (40.0, True)


def test_unknown_bonus_mode_raises() -> None:
    # Act and Assert
    with pytest.raises(ValueError, match="bonus_mode"):
        GPWrapper(HUGIN(), bonus_mode="often")


def test_gas_summary_reports_the_least_finished_patch_and_the_nearest_unfinished_cell() -> None:
    # Arrange: a finished patch and a patch with half of its cells known
    base, gp, _ = make_bonus_env()
    base.GT_c_around_threshold_maps[25:27, 25:29] = True  # 8 cells
    gp.GP_std[25:27, 25:27] = 0.05

    # Act: from cell (25, 20) to the first unknown cell of the patch, (25, 27)
    lowest_share, nearest = gp.gas_summary(25, 20)

    # Assert
    assert lowest_share == pytest.approx(0.5)
    assert nearest == 7.0


def test_gas_summary_with_no_gas_found() -> None:
    # Arrange
    _, gp = make_meta_env()
    gp.reset()

    # Act and Assert: nothing is waiting
    assert gp.gas_summary(10, 10) == (1.0, float("inf"))
