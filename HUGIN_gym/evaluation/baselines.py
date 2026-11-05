"""Meta agents without learning, to compare the trained meta agents with. None of them gets the number of plumes or the true map."""

from collections.abc import Callable

import numpy as np

BASELINES = ("lawnmower", "space", "rule", "random")


class ScriptedMeta:
    """Same call as a PPO model, so run_episode_hrl needs no change."""

    def __init__(self, choose: Callable[[], int]) -> None:
        self.choose = choose

    def predict(self, obs: dict, deterministic: bool = True) -> tuple[int, None]:
        return self.choose(), None


class RuleChooser:
    """A simple rule from what the GP has found so far, as a benchmark a trained meta agent must beat.

    The first option (sweep or space) runs unless a found plume is unfinished. Then the gas sub-agent with the lower
    share runs. After a decision without progress the other gas sub-agent gets a turn, and after two the first option.
    """

    def __init__(self, env) -> None:
        self.env = env
        self.stuck = 0
        self.last_choice = 0
        self.last_progress: tuple[int, ...] | None = None

    def __call__(self) -> int:
        env = self.env
        base = env.base_env.unwrapped
        progress = tuple(
            int(np.sum(base.state[key])) for key in ("visited_map", "c_over_threshold_map", "c_around_threshold_map")
        )
        self.stuck = self.stuck + 1 if env.primitive_steps > 0 and progress == self.last_progress else 0
        self.last_progress = progress

        plume_share, border_share = env.base_env.goal_shares(env.current_obs)
        plume_found = base.maxN_over_thresh > env.base_env.N_corners
        unfinished = plume_found and (plume_share < env.accuracy_goals[1] or border_share < env.accuracy_goals[2])
        if not unfinished or self.stuck >= 2:
            choice = 0
        elif self.stuck == 1 and self.last_choice in (1, 2):
            choice = 3 - self.last_choice
        else:
            choice = 1 if plume_share < border_share else 2
        self.last_choice = choice
        return choice


def make_baseline(name: str, env, seed: int) -> ScriptedMeta:
    """The baseline called name, for an env whose options are in the order space or sweep, plume, border."""
    match name:
        case "lawnmower" | "space":  # always the first option: a sweep with --spacing, otherwise the space sub-agent
            return ScriptedMeta(lambda: 0)
        case "rule":
            return ScriptedMeta(RuleChooser(env))
        case "random":
            rng = np.random.default_rng(seed)  # not np.random: the env draws its episodes from there
            return ScriptedMeta(lambda: int(rng.integers(env.action_space.n)))
        case _:
            raise ValueError(f"Unknown baseline {name!r}, choose from {BASELINES}")
