"""Helpers for the estimated gas maps, shared by the GP wrapper, the options and the evaluation scripts."""

import numpy as np


def gp_maps(gp_mean: np.ndarray, c_threshold: float, c_width: float) -> tuple[np.ndarray, np.ndarray]:
    """The plume and border cells of the GP's estimate, from its mean concentration."""
    plume = gp_mean > c_threshold
    border = (gp_mean > c_threshold - c_width) & (gp_mean <= c_threshold + c_width)
    return plume, border


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Cells in both maps divided by cells in either map (intersection over union). 1.0 means identical maps."""
    union = np.sum(a | b)
    return float(np.sum(a & b) / union) if union else 1.0


def unfinished_cells(
    belief: np.ndarray, corners: np.ndarray, visited: np.ndarray, std: np.ndarray, goal_mode: str, known_std: float
) -> np.ndarray:
    """The cells of an estimated map that are not finished, as the end goal counts them. All maps have shape (X, Y)."""
    cells = belief & ~corners
    return cells & ~visited if goal_mode == "visit" else cells & (std > known_std)


def patch_mask(shape: tuple[int, int], ix: int, iy: int, radius: int) -> np.ndarray:
    """The cells inside the square patch around (ix, iy): what a sub-agent sees of the map."""
    mask = np.zeros(shape, dtype=bool)
    mask[max(ix - radius, 0) : ix + radius + 1, max(iy - radius, 0) : iy + radius + 1] = True
    return mask


def nearest_distance(cells: np.ndarray, ix: int, iy: int) -> float:
    """The distance in cells from (ix, iy) to the nearest marked cell, infinite if there is none."""
    xs, ys = np.nonzero(cells)
    return float(np.sqrt(np.min((xs - ix) ** 2 + (ys - iy) ** 2))) if len(xs) else float("inf")
