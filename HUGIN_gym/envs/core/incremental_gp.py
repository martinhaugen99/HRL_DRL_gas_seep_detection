"""Exact Gaussian process with a fixed RBF kernel, updated one measurement at a time."""

import numpy as np
from scipy.linalg import solve_triangular


class IncrementalGP:
    """The mean and std of sklearn's GaussianProcessRegressor with the fixed kernel
    C(1.0) * RBF(length_scale) + WhiteKernel(noise), without refitting from scratch.

    For every new measurement sklearn recomputes the Cholesky factor L of K(X, X) and V = L^-1 K(X, grid). A new
    measurement only appends one row to L and one row to V, so with n measurements and N grid cells an update costs
    about n^2 + n * N operations, against n^3 + n^2 * N for a refit.
    """

    def __init__(
        self,
        grid: np.ndarray,
        length_scale: float = 3.5,
        noise: float = 1e-4,
        jitter: float = 1e-10,  # sklearn's default alpha
        capacity: int = 512,
    ) -> None:
        self.grid = np.asarray(grid, dtype=float)
        self.two_length_scale_sq = 2 * length_scale**2
        self.prior_var = 1.0 + noise
        self.diagonal = 1.0 + noise + jitter
        self.L = np.zeros((capacity, capacity))
        self.V = np.zeros((capacity, len(self.grid)))
        self.z = np.zeros(capacity)  # L^-1 y
        self.X = np.zeros((capacity, self.grid.shape[1]))
        self.reset()

    def reset(self) -> None:
        """Forget all measurements. The arrays are kept and reused."""
        self.n = 0
        self.mean = np.zeros(len(self.grid))
        self.var = np.full(len(self.grid), self.prior_var)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 0.0))

    def add(self, position: np.ndarray, value: float) -> None:
        """Add one measurement and update the mean and variance of every grid cell."""
        n = self.n
        if n == len(self.z):
            self._grow()
        position = np.asarray(position, dtype=float)
        to_grid = self._rbf(self.grid, position)
        if n:
            row = solve_triangular(self.L[:n, :n], self._rbf(self.X[:n], position), lower=True, check_finite=False)
            pivot = np.sqrt(self.diagonal - row @ row)
            v = (to_grid - row @ self.V[:n]) / pivot
            z = (value - row @ self.z[:n]) / pivot
            self.L[n, :n] = row
        else:
            pivot = np.sqrt(self.diagonal)
            v = to_grid / pivot
            z = value / pivot
        self.L[n, n] = pivot
        self.V[n] = v
        self.z[n] = z
        self.X[n] = position
        self.n = n + 1
        # new arrays, not in place: callers may still hold the previous mean
        self.mean = self.mean + v * z
        self.var = self.var - v * v

    def _rbf(self, points: np.ndarray, position: np.ndarray) -> np.ndarray:
        return np.exp(-np.sum((points - position) ** 2, axis=1) / self.two_length_scale_sq)

    def _grow(self) -> None:
        n = len(self.z)
        L = np.zeros((2 * n, 2 * n))
        L[:n, :n] = self.L
        self.L = L
        self.V = np.vstack([self.V, np.zeros_like(self.V)])
        self.z = np.concatenate([self.z, np.zeros(n)])
        self.X = np.vstack([self.X, np.zeros_like(self.X)])
