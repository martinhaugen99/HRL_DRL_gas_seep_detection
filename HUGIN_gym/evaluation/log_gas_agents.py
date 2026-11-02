"""Log what the plume and border agents see and do inside a survey, to find out why they idle.

Alone, in their own training setup, these agents do well. Inside a survey they often make no progress. For every decision
that uses one of them, this script records what the agent could see when it started and what it then did:
- its target: the cells of its own map (plume or border estimate) that are not finished, as the end goal counts them
- whether a target cell is inside the 7 x 7 patch around the AUV, which is all the agent sees of the map
- its other inputs: how much of the map is visited, and how much of the patch is visited
- its moves, edge hits and how many target cells the decision finished

Run from the repository root, e.g.:
    PYTHONPATH=. .venv/bin/python -m HUGIN_gym.evaluation.log_gas_agents \
        --baseline rule --spacing 4 --sources 1 3 --episodes 300 --output ../gas_agents.csv
"""

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
from stable_baselines3.common.utils import set_random_seed

from HUGIN_gym.evaluation.compare_maps import evaluation

OPTION_NAMES = {1: "plume", 2: "border"}
ACTION_NAMES = {0: "forward", 1: "up", 2: "down", 3: "left", 4: "right"}
DISTANCE_BINS = (
    (0.0, 3.0, "up to 3 cells"),
    (3.0, 6.0, "3 to 6 cells"),
    (6.0, 10.0, "6 to 10 cells"),
    (10.0, np.inf, "more than 10 cells"),
)
COVERAGE_BINS = (
    (0.0, 0.15, "up to 15 %"),
    (0.15, 0.3, "15 to 30 %"),
    (0.3, 0.5, "30 to 50 %"),
    (0.5, 1.01, "over 50 %"),
)


def unfinished_cells(
    belief: np.ndarray, corners: np.ndarray, visited: np.ndarray, std: np.ndarray, goal_mode: str, known_std: float
) -> np.ndarray:
    """The cells of an estimated map that are not finished, as the end goal counts them. All maps have shape (X, Y)."""
    cells = belief & ~corners
    return cells & ~visited if goal_mode == "visit" else cells & (std > known_std)


def patch_mask(shape: tuple[int, int], ix: int, iy: int, radius: int) -> np.ndarray:
    """The cells inside the square patch around (ix, iy): what the agent sees of the map."""
    mask = np.zeros(shape, dtype=bool)
    mask[max(ix - radius, 0) : ix + radius + 1, max(iy - radius, 0) : iy + radius + 1] = True
    return mask


def nearest_distance(cells: np.ndarray, ix: int, iy: int) -> float:
    """The distance in cells from (ix, iy) to the nearest marked cell, infinite if there is none."""
    xs, ys = np.nonzero(cells)
    return float(np.sqrt(np.min((xs - ix) ** 2 + (ys - iy) ** 2))) if len(xs) else float("inf")


def own_belief(base, option: int) -> np.ndarray:
    """The agent's own estimated map: the plume for the plume agent, the border for the border agent."""
    return (base.GT_c_over_threshold_maps if option == 1 else base.GT_c_around_threshold_maps)[:, :, 0]


def run_episode(env, model, deterministic: bool, episode: int) -> list[dict]:
    """Run one episode and return one row per decision of the plume or border agent."""
    gp_env = env.base_env
    base = gp_env.unwrapped
    corners = gp_env.corner_mask[:, :, 0]
    step_log: list[tuple[int, bool]] = []
    original_step = gp_env.step

    def logged_step(action):
        result = original_step(action)
        step_log.append((int(action), result[1] == -4.0))
        return result

    gp_env.step = logged_step
    obs, _ = env.reset()
    rows: list[dict] = []
    poses_in_gas_streak: set[tuple] = set()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        option, _length = env._decode_action(int(action))
        if option not in OPTION_NAMES:
            poses_in_gas_streak.clear()
            step_log.clear()
            obs, _, terminated, truncated, _ = env.step(int(action))
            continue

        current = env.current_obs
        visited_before = base.state["visited_map"][:, :, 0, 0].copy()
        unfinished_before = unfinished_cells(
            own_belief(base, option), corners, visited_before, gp_env.GP_std, gp_env.goal_mode, gp_env.known_std
        )
        known_before = gp_env.GP_std <= gp_env.known_std
        ix, iy, _iz = base._pos_to_idx(base.state["x"], base.state["y"], base.state["z"])
        patch = patch_mask(unfinished_before.shape, ix, iy, base.patch_radius)
        progress_before = tuple(
            int(np.sum(base.state[key])) for key in ("visited_map", "c_over_threshold_map", "c_around_threshold_map")
        )
        row = {
            "episode": episode,
            "option": OPTION_NAMES[option],
            "survey_step": env.primitive_steps,
            "space_coverage": float(current["space_coverage"][0]),
            "own_coverage": float(current["plume_coverage" if option == 1 else "border_coverage"][0]),
            "unfinished_total": int(unfinished_before.sum()),
            "unfinished_in_patch": int((unfinished_before & patch).sum()),
            "gas_in_patch": int((own_belief(base, option) & ~corners & patch).sum()),
            "nearest_unfinished": nearest_distance(unfinished_before, ix, iy),
            "visited_in_patch": float(visited_before[patch].mean()),
        }

        step_log.clear()
        obs, _, terminated, truncated, _ = env.step(int(action))

        visited_after = base.state["visited_map"][:, :, 0, 0]
        belief_after = own_belief(base, option) & ~corners
        if gp_env.goal_mode == "visit":
            finished_now = belief_after & visited_after & ~visited_before
        else:
            finished_now = belief_after & (gp_env.GP_std <= gp_env.known_std) & ~known_before
        pose = (base.state["x"], base.state["y"], round(float(base.state["theta"]), 3))
        progress_after = tuple(
            int(np.sum(base.state[key])) for key in ("visited_map", "c_over_threshold_map", "c_around_threshold_map")
        )
        row.update(
            actions=" ".join(ACTION_NAMES[action] for action, _ in step_log),
            steps=len(step_log),
            edge_hits=sum(hit for _, hit in step_log),
            new_cells=int(visited_after.sum() - visited_before.sum()),
            target_cells_finished=int(finished_now.sum()),
            idle=progress_after == progress_before,
            repeated_pose=pose in poses_in_gas_streak,
        )
        poses_in_gas_streak.add(pose)
        rows.append(row)
    gp_env.step = original_step
    for row in rows:
        row["episode_ended_by_goal"] = bool(terminated)
    return rows


def _rates(rows: list[dict]) -> str:
    """The share of decisions that finished a target cell, idled, moved without finishing one, hit the edge or came back to a pose."""
    n = len(rows)
    if n == 0:
        return "no decisions"
    useful = sum(row["target_cells_finished"] > 0 for row in rows)
    idle = sum(row["idle"] for row in rows)
    return (
        f"{n:6d} decisions | finished a target cell {100 * useful / n:5.1f} % | idle {100 * idle / n:5.1f} % | "
        f"moved without finishing one {100 * (n - useful - idle) / n:5.1f} % | edge hit {100 * sum(row['edge_hits'] > 0 for row in rows) / n:5.1f} % | "
        f"back to an earlier pose {100 * sum(row['repeated_pose'] for row in rows) / n:5.1f} %"
    )


def _in_bin(key: str, low: float, high: float) -> Callable[[dict], bool]:
    return lambda row: low <= row[key] < high


def _table(rows: list[dict], title: str, groups: list[tuple[str, Callable[[dict], bool]]]) -> list[str]:
    lines = [title]
    for label, test in groups:
        group = [row for row in rows if test(row)]
        lines.append(f"  {label:>32}: share {100 * len(group) / max(len(rows), 1):5.1f} % | {_rates(group)}")
    return lines


def summarise(rows: list[dict]) -> str:
    """The summary of all decisions, per agent."""
    lines: list[str] = []
    for option in OPTION_NAMES.values():
        own = [row for row in rows if row["option"] == option]
        lines.append(f"=== {option} agent: {_rates(own)}")
        if not own:
            continue
        lines += _table(
            own,
            "By what it can see when the decision starts:",
            [
                ("target cell in its patch", lambda row: row["unfinished_in_patch"] > 0),
                (
                    "own gas in patch, all finished",
                    lambda row: row["unfinished_in_patch"] == 0 and row["gas_in_patch"] > 0,
                ),
                ("no gas of its own in patch", lambda row: row["gas_in_patch"] == 0 and row["unfinished_total"] > 0),
                ("no target left on the map", lambda row: row["unfinished_total"] == 0),
            ],
        )
        lines += _table(
            own,
            "By the distance to the nearest target cell:",
            [(label, _in_bin("nearest_unfinished", low, high)) for low, high, label in DISTANCE_BINS]
            + [("no target left", lambda row: row["nearest_unfinished"] == float("inf"))],
        )
        sees_target = [row for row in own if row["unfinished_in_patch"] > 0]
        lines += _table(
            sees_target,
            "With a target cell in its patch, by the share of the map visited:",
            [(label, _in_bin("space_coverage", low, high)) for low, high, label in COVERAGE_BINS],
        )
        lines += _table(
            sees_target,
            "With a target cell in its patch, by the share of the patch visited:",
            [(label, _in_bin("visited_in_patch", low, high)) for low, high, label in COVERAGE_BINS],
        )
        for label, group in (
            ("finished a target cell", [row for row in own if row["target_cells_finished"] > 0]),
            ("finished none", [row for row in own if row["target_cells_finished"] == 0]),
        ):
            moves = " ".join(move for row in group for move in row["actions"].split())
            counts = {name: moves.split().count(name) for name in ACTION_NAMES.values()}
            total = max(sum(counts.values()), 1)
            shares = ", ".join(f"{name} {100 * count / total:.0f} %" for name, count in counts.items())
            lines.append(f"  moves when it {label}: {shares}")
        lines.append("")
    ended = [row for row in rows if row["episode_ended_by_goal"]]
    not_ended = [row for row in rows if not row["episode_ended_by_goal"]]
    n_episodes = {
        flag: len({row["episode"] for row in rows if row["episode_ended_by_goal"] == flag}) for flag in (True, False)
    }
    lines.append("By how the episode ended (episodes with at least one such decision):")
    lines.append(f"  ended by the goal, {n_episodes[True]} episodes: {_rates(ended)}")
    lines.append(f"  budget used up, {n_episodes[False]} episodes: {_rates(not_ended)}")
    for flag, group in ((True, ended), (False, not_ended)):
        if n_episodes[flag]:
            label = "ended by the goal" if flag else "budget used up"
            lines.append(
                f"  AUV steps per episode in these agents, {label}: {sum(row['steps'] for row in group) / n_episodes[flag]:.0f}"
            )
    return "\n".join(lines)


def save_rows(rows: list[dict], path: Path) -> None:
    """Save one row per decision as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line of the evaluation script, plus the output file."""
    parser = evaluation.make_parser()
    parser.description = "Log what the plume and border agents see and do inside a survey."
    parser.add_argument(
        "--output", type=Path, help="CSV file with one row per decision (default: only print the summary)"
    )
    args = parser.parse_args(argv)
    if args.mode not in (None, "1", "4"):
        parser.error("--mode must be 1 (deterministic) or 4 (sampled, the default, as in training)")
    if args.scripted_gas:
        parser.error("--scripted-gas replaces the agents this script logs")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    survey = evaluation.make_survey(args.model, **evaluation.survey_settings(args))
    set_random_seed(evaluation.SEED)
    rows = [
        row
        for episode in range(args.episodes)
        for row in run_episode(survey.env, survey.model, args.mode == "1", episode)
    ]
    survey.env.close()
    print(summarise(rows))
    if args.output:
        save_rows(rows, args.output)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
