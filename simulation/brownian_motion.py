import numpy as np
from typing import Union
from numpy.typing import NDArray
from numpy import float_

from utility import DEFAULT_SEED, to_numpy


def simulate_brownian_motion_from_increments(
    size: int,
    t_grid: Union[float, NDArray[float_]],
    dim: int = 1,
    rng: np.random.Generator = None
) -> NDArray[float_]:
    """
    Simulates the trajectory of the standard d-dimensional Brownian motion with the increments method.

    Args:
        t_grid: time grid to simulate the price on.
        size: number of simulated trajectories.
        dim: dimensionality of the Brownian motion.
        rng: `np.random.Generator` used for simulation.

    Returns:
        np.ndarray of shape (size, dim, len(t_grid)) with simulated trajectories if model dimension greater than 1.
    """
    if rng is None:
        rng = np.random.default_rng(seed=DEFAULT_SEED)
    t_grid = to_numpy(t_grid)
    dt = np.diff(t_grid)
    if np.any(dt < 0):
        raise ValueError("Time grid should be increasing.")

    dW = rng.normal(size=(size, dim, len(dt)))
    return np.concatenate([np.zeros((size, dim, 1)), np.cumsum(np.sqrt(dt) * dW, axis=2)], axis=2)


def simulate_brownian_motion_from_brownian_bridge(
    size: int,
    t_grid: Union[float, NDArray[float_]],
    dim: int = 1,
    rng: np.random.Generator = None
) -> NDArray[float_]:
    """
    Simulates the trajectory of the standart d-dimensional Brownian motion using the Brownian bridge.

    Args:
        t_grid: time grid to simulate the price on.
        size: number of simulated trajectories.
        dim: dimensionality of the Brownian motion.
        rng: `np.random.Generator` used for simulation.

    Returns:
        np.ndarray of shape (size, dim, len(t_grid)) with simulated trajectories if model dimension greater than 1.
    """
    if rng is None:
        rng = np.random.default_rng(seed=DEFAULT_SEED)

    t_grid = to_numpy(t_grid)
    dt = np.diff(t_grid)
    if np.any(dt < 0):
        raise ValueError("Time grid should be increasing.")

    if t_grid.shape[0] == 1:
        return np.zeros((size, dim, 1))

    standard_normal_sample = rng.normal(size=(size, dim, t_grid.shape[0]))

    W = np.zeros((size, dim, t_grid.shape[0]))
    W[:, :, -1] = np.sqrt(t_grid[-1]) * standard_normal_sample[:, :, -1]

    # indicator of filled in W
    map = np.zeros(t_grid.shape[0])
    map[0] = 1
    map[-1] = 1

    for _ in range(int(np.log2(t_grid.size)+1)):
        idx_filled,  = np.where(map == 1)
        idx_left = idx_filled[:-1]
        idx_right = idx_filled[1:]
        idx_center = (idx_left + idx_right) // 2
        # choose indices not in a row
        idx_to_complete = idx_center != idx_left
        idx_center = idx_center[idx_to_complete]
        idx_left = idx_left[idx_to_complete]
        idx_right = idx_right[idx_to_complete]

        map[idx_center] = 1

        E_step = ((t_grid[idx_right] - t_grid[idx_center]) / (t_grid[idx_right] - t_grid[idx_left]) * W[:, :, idx_left] +
                  (t_grid[idx_center] - t_grid[idx_left]) / (t_grid[idx_right] - t_grid[idx_left]) * W[:, :, idx_right])
        variance_step = ((t_grid[idx_center] - t_grid[idx_left]) * (t_grid[idx_right] - t_grid[idx_center]) / (t_grid[idx_right] - t_grid[idx_left]))
        W[:, :, idx_center] = E_step + np.sqrt(variance_step) * standard_normal_sample[:, :, idx_center]
    return W
