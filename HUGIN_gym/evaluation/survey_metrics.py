"""How good is the map at the end of a survey? The same metrics for trained agents and baselines."""

import numpy as np
from scipy import ndimage

from HUGIN_gym.envs.core.gas_cells import gp_maps, overlap


def share(part: np.ndarray, whole: np.ndarray) -> float:
    """Share of the cells of whole that are also in part. 1.0 if whole is empty."""
    return float(np.sum(part & whole) / np.sum(whole)) if whole.any() else 1.0


def map_metrics(
    gp_mean: np.ndarray,
    true_over: np.ndarray,
    true_around: np.ndarray,
    visited: np.ndarray,
    c_threshold: float,
    c_width: float,
    visited_goal: float = 0.9,
) -> dict[str, float]:
    """Compare the GP's map with the true map at the end of an episode. All arguments are maps of shape (X, Y).

    A true plume is a connected region of cells above the threshold, so sources that overlap count as one plume.
    """
    gp_over, gp_around = gp_maps(gp_mean, c_threshold, c_width)
    labels, n_plumes = ndimage.label(true_over)
    plumes = [labels == i + 1 for i in range(n_plumes)]
    return {
        "plume_overlap": overlap(gp_over, true_over),
        "border_overlap": overlap(gp_around, true_around),
        "true_plume_visited": share(visited, true_over),
        "true_border_visited": share(visited, true_around),
        "plumes": n_plumes,
        "plumes_found": sum(bool(np.any(gp_over & plume)) for plume in plumes),
        "plumes_visited": sum(share(visited, plume) >= visited_goal for plume in plumes),
    }


def summarise_by_plumes(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """One row per number of true plumes: how many episodes, and the mean of each value over those episodes."""
    summary = []
    for n_plumes in sorted({row["plumes"] for row in rows}):
        group = [row for row in rows if row["plumes"] == n_plumes]
        means = {key: float(np.mean([row[key] for row in group])) for key in group[0]}
        summary.append({**means, "plumes": n_plumes, "episodes": len(group)})
    return summary
