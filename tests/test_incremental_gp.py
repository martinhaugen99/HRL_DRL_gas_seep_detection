"""Tests that IncrementalGP gives the same mean and std as sklearn's GP with the same fixed kernel."""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from HUGIN_gym.envs.core.incremental_gp import IncrementalGP

COORDS = np.arange(-6.0, 7.0)
GRID = np.column_stack([axis.ravel() for axis in np.meshgrid(COORDS, COORDS)])


def sklearn_gp(X: np.ndarray, y: np.ndarray, length_scale: float, noise: float) -> tuple[np.ndarray, np.ndarray]:
    kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(
        noise_level=noise, noise_level_bounds=(1e-8, 1e1)
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, optimizer=None)
    gp.fit(X, y)
    return gp.predict(GRID, return_std=True)


def path_with_repeats(rng: np.random.Generator, n: int) -> np.ndarray:
    """A random walk on the grid, so positions repeat as they do when the AUV crosses its own path."""
    steps = rng.integers(-1, 2, size=(n, 2))
    return np.clip(np.cumsum(steps, axis=0), -6, 6).astype(float)


@pytest.mark.parametrize(("length_scale", "noise"), [(3.5, 1e-4), (2.0, 1e-3)])
def test_matches_sklearn_after_every_measurement(length_scale: float, noise: float) -> None:
    # Arrange
    rng = np.random.default_rng(0)
    X = path_with_repeats(rng, 40)
    y = rng.uniform(0.0, 1.0, size=len(X))
    gp = IncrementalGP(GRID, length_scale=length_scale, noise=noise)

    for i in range(len(X)):
        # Act
        gp.add(X[i], y[i])

        # Assert
        mean, std = sklearn_gp(X[: i + 1], y[: i + 1], length_scale, noise)
        np.testing.assert_allclose(gp.mean, mean, atol=1e-9)
        np.testing.assert_allclose(gp.std, std, atol=1e-7)


def test_grows_past_its_capacity() -> None:
    # Arrange
    rng = np.random.default_rng(1)
    X = path_with_repeats(rng, 20)
    y = rng.uniform(0.0, 1.0, size=len(X))
    gp = IncrementalGP(GRID, capacity=4)

    # Act
    for position, value in zip(X, y, strict=True):
        gp.add(position, value)

    # Assert
    mean, _ = sklearn_gp(X, y, 3.5, 1e-4)
    np.testing.assert_allclose(gp.mean, mean, atol=1e-9)


def test_reset_forgets_the_measurements() -> None:
    # Arrange
    gp = IncrementalGP(GRID)
    gp.add(np.array([0.0, 0.0]), 1.0)

    # Act
    gp.reset()
    gp.add(np.array([1.0, 1.0]), 0.5)

    # Assert
    mean, std = sklearn_gp(np.array([[1.0, 1.0]]), np.array([0.5]), 3.5, 1e-4)
    np.testing.assert_allclose(gp.mean, mean, atol=1e-9)
    np.testing.assert_allclose(gp.std, std, atol=1e-7)


def test_previous_mean_is_not_changed_by_a_new_measurement() -> None:
    # Arrange
    gp = IncrementalGP(GRID)
    gp.add(np.array([0.0, 0.0]), 1.0)
    previous = gp.mean
    copy = previous.copy()

    # Act
    gp.add(np.array([1.0, 1.0]), 0.5)

    # Assert
    np.testing.assert_array_equal(previous, copy)
