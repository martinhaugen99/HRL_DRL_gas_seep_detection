from collections import deque

import numpy as np
import torch as th
from scipy.ndimage import binary_dilation

FORWARD, LEFT, RIGHT = 0, 3, 4  # the actions that move the AUV on the 2D map; up and down always hit the edge there
QUARTER_TURN = np.pi / 2
KNOWN_STD = 0.05  # a sweep waypoint with a lower GP std is already measured, so the sweep skips it


class Option:
    def __init__(self, policy, termination_fn, name="option", obs_keys=None,obs_transform=None):
        self.policy = policy
        self.termination_fn = termination_fn
        self.name = name
        self.obs_keys = obs_keys
        self.obs_transform = obs_transform

    def _filter_obs(self, obs):
        if self.obs_keys is None:
            return obs
        return {k: obs[k] for k in self.obs_keys}

    def act(self, state):
        filtered_state = self._filter_obs(state)
        if self.obs_transform is not None:
            filtered_state = self.obs_transform(filtered_state)
        action, _ = self.policy.predict(filtered_state, deterministic=True)
        return action

    def value(self, state) -> float:
        """The sub-agent's own estimate of its return from this state (its PPO critic)."""
        filtered_state = self._filter_obs(state)
        if self.obs_transform is not None:
            filtered_state = self.obs_transform(filtered_state)
        obs_tensor, _ = self.policy.policy.obs_to_tensor(filtered_state)
        with th.no_grad():
            return float(self.policy.policy.predict_values(obs_tensor).item())

    def should_terminate(self, state, next_state, threshold1=None):
        state_f = self._filter_obs(state)
        next_state_f = self._filter_obs(next_state)
        return self.termination_fn(state_f, next_state_f, threshold1=threshold1)


class ScriptedOption:
    """A scripted behavior with the interface of Option, so the meta agent can choose it like a sub-agent."""

    def __init__(self, base_env, gp_env, name: str) -> None:
        if base_env.map_size_z > 1:
            raise ValueError("Scripted options only work on the 2D map")
        self.base_env = base_env
        self.gp_env = gp_env
        self.name = name
        self.limit = int(base_env.domain_limit["x"])
        self.moves = self._moves_per_heading(base_env.move_vectors)

    @staticmethod
    def _moves_per_heading(move_vectors: dict) -> list[list[tuple[int, int, int, int]]]:
        """(action, dx, dy, new heading) per heading (0: +x, 1: +y, 2: -x, 3: -y), as in Dynamics.step."""
        moves = []
        for heading in range(4):
            theta = heading * QUARTER_TURN
            per_action = []
            for action in (FORWARD, LEFT, RIGHT):
                dx_local, dy_local, _, dtheta = move_vectors[action]
                dx = round(dx_local * np.cos(theta) - dy_local * np.sin(theta))
                dy = round(dx_local * np.sin(theta) + dy_local * np.cos(theta))
                per_action.append((action, dx, dy, (heading + round(dtheta / QUARTER_TURN)) % 4))
            moves.append(per_action)
        return moves

    def _pose(self) -> tuple[int, int, int]:
        state = self.base_env.state
        return int(state["x"]), int(state["y"]), round((state["theta"] % (2 * np.pi)) / QUARTER_TURN) % 4

    def _is_legal(self, x: int, y: int, heading: int) -> bool:
        """False for the poses HUGIN._check_boundary_violation rejects."""
        limit = self.limit
        if abs(x) > limit or abs(y) > limit:
            return False
        return not ((x == limit and heading == 0) or (x == -limit and heading == 2) or (y == limit and heading == 1) or (y == -limit and heading == 3))

    def _first_action_to(self, x: int, y: int, heading: int, targets: np.ndarray) -> int:
        """First action of a shortest path (breadth-first search over x, y and heading) to within 1 cell of a target."""
        near_target = binary_dilation(targets, structure=np.ones((3, 3), dtype=bool))
        start = (x, y, heading)
        first_action: dict[tuple[int, int, int], int | None] = {start: None}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for action, dx, dy, new_heading in self.moves[state[2]]:
                new_state = (state[0] + dx, state[1] + dy, new_heading)
                if new_state in first_action or not self._is_legal(*new_state):
                    continue
                previous = first_action[state]
                first = action if previous is None else previous
                if near_target[new_state[0] + self.limit, new_state[1] + self.limit]:
                    return first
                first_action[new_state] = first
                queue.append(new_state)
        return FORWARD

    def value(self, obs: dict) -> float:
        """A scripted option has no critic. Needed for the meta states with sub-agent values."""
        return 0.0


class SweepOption(ScriptedOption):
    """A scripted lawnmower.

    The tracks are spacing cells apart and start at the corner the AUV starts in. After another option has moved the
    AUV, the sweep goes on with the next waypoint it has not reached. When the sweep is done, the AUV goes to the part
    of the map the GP knows least. Always choosing this option gives a plain lawnmower.
    """

    def __init__(self, base_env, gp_env, spacing: int, name: str = "sweep") -> None:
        if spacing < 1:
            raise ValueError(f"spacing must be at least 1, not {spacing}")
        super().__init__(base_env, gp_env, name)
        self.spacing = spacing
        self.waypoints: deque[tuple[int, int]] = deque()

    def reset(self) -> None:
        """Plan the sweep from the AUV's start: tracks along its heading, from the side of the map it starts on."""
        x, y, heading = self._pose()
        along_y = heading % 2 == 1
        across, along = (x, y) if along_y else (y, x)
        last_cell = self.limit - 1
        across_sign, along_sign = (1 if across < 0 else -1), (1 if along < 0 else -1)
        across_start, along_start = -across_sign * last_cell, -along_sign * last_cell

        self.waypoints.clear()
        forward = True
        for track in range(across_start, -across_start + across_sign, across_sign * self.spacing):
            first, sign = (along_start, along_sign) if forward else (-along_start, -along_sign)
            for cell in range(first, -first + sign, sign * 2):  # the forward action moves 2 cells
                self.waypoints.append((track, cell) if along_y else (cell, track))
            forward = not forward

    def act(self, obs: dict) -> int:
        x, y, heading = self._pose()
        std = self.gp_env.GP_std
        while self.waypoints and self._is_done(self.waypoints[0], x, y, std):
            self.waypoints.popleft()
        if self.waypoints:
            targets = np.zeros_like(std, dtype=bool)
            targets[self.waypoints[0][0] + self.limit, self.waypoints[0][1] + self.limit] = True
        else:
            targets = std >= np.percentile(std, 99)
        return self._first_action_to(x, y, heading, targets)

    def _is_done(self, waypoint: tuple[int, int], x: int, y: int, std: np.ndarray) -> bool:
        reached = abs(waypoint[0] - x) <= 1 and abs(waypoint[1] - y) <= 1
        return reached or std[waypoint[0] + self.limit, waypoint[1] + self.limit] < KNOWN_STD


class GasFillOption(ScriptedOption):
    """Scripted gas handling, as a benchmark for the plume and border agents.

    The AUV takes the shortest path to the nearest unfinished cell of the estimated plume or border. Unfinished follows
    the end goal of GPWrapper: not visited with goal_mode "visit", and a GP std above known_std with "estimate". When
    nothing is unfinished, the AUV goes to the part of the map the GP knows least.
    """

    def __init__(self, base_env, gp_env, name: str = "scripted gas fill") -> None:
        super().__init__(base_env, gp_env, name)

    def act(self, obs: dict) -> int:
        base = self.base_env
        corners = self.gp_env.corner_mask[:, :, 0]
        gas = (base.GT_c_over_threshold_maps | base.GT_c_around_threshold_maps)[:, :, 0] & ~corners
        std = self.gp_env.GP_std
        if self.gp_env.goal_mode == "visit":
            unfinished = gas & ~base.state["visited_map"][:, :, 0, 0]
        else:
            unfinished = gas & (std > self.gp_env.known_std)
        targets = unfinished if unfinished.any() else std >= np.percentile(std, 99)
        return self._first_action_to(*self._pose(), targets)
