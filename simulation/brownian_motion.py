import numpy as np
from typing import Union
from numpy.typing import NDArray
from numpy import float_

from simulation.utility import DEFAULT_SEED


def simulate_brownian_motion_from_increments(
    size: int,
    t_grid: Union[float, NDArray[float_]],
    dim: int = 1,
    rng: np.random.Generator = None
) -> Union[float, NDArray[float_]]:
    """
    Simulates the trajectory of the standard d-dimensional Brownian motion with the increments method.

    Args:
        t_grid: time grid to simulate the price on.
        size: number of simulated trajectories.
        dim: dimensionality of the Brownian motion.
        rng: `np.random.Generator` used for simulation.

    Returns:
        np.ndarray of shape (size, len(t_grid)) with simulated trajectories if model dimension is 1.
        np.ndarray of shape (size, dim, len(t_grid)) with simulated trajectories if model dimension greater than 1.
    """
    if rng is None:
        rng = np.random.default_rng(seed=DEFAULT_SEED)
    if isinstance(t_grid, list):
        t_grid = np.array(t_grid)
    if not isinstance(t_grid, np.ndarray):
        t_grid = np.array([t_grid])
    dt = np.diff(np.concatenate([np.zeros(1), t_grid]))
    if np.any(dt < 0):
        raise ValueError("Time grid should be increasing.")

    dW = rng.normal(size=(size, dim, len(t_grid)))
    if dim == 1:
        return np.cumsum(np.sqrt(dt) * dW, axis=2)[:, 0, :]
    return np.cumsum(np.sqrt(dt) * dW, axis=2)


def simulate_brownian_motion_from_brownian_bridge(
    size: int,
    t_grid: Union[float, NDArray[float_]],
    dim: int = 1,
    rng: np.random.Generator = None
) -> Union[float, NDArray[float_]]:
    """
    Simulates the trajectory of the standart d-dimensional Brownian motion using the Brownian bridge.

    Args:
        t_grid: time grid to simulate the price on.
        size: number of simulated trajectories.
        dim: dimensionality of the Brownian motion.
        rng: `np.random.Generator` used for simulation.

    Returns:
        np.ndarray of shape (size, len(t_grid)) with simulated trajectories if model dimension is 1.
        np.ndarray of shape (size, dim, len(t_grid)) with simulated trajectories if model dimension greater than 1.
    """
    if rng is None:
        rng = np.random.default_rng(seed=DEFAULT_SEED)

    if isinstance(t_grid, list):
        t_grid = np.array(t_grid)
    if not isinstance(t_grid, np.ndarray):
        t_grid = np.array([t_grid])
    dt = np.diff(np.concatenate([np.zeros(1), t_grid]))
    if np.any(dt < 0):
        raise ValueError("Time grid should be increasing.")

    if t_grid.shape[0] == 1:
        dW = rng.normal(size=(size, dim, len(t_grid)))
        if dim == 1:
            return np.cumsum(np.sqrt(dt) * dW, axis=2)[:, 0, :]
        return np.cumsum(np.sqrt(dt) * dW, axis=2)

    W = np.zeros((size, dim, t_grid.shape[0]))
    W[:, :, -1] = np.sqrt(t_grid[-1]) * rng.normal(size=(size, dim))

    map = np.zeros(t_grid.shape[0])
    map[0] = 1
    map[-1] = 1

    while sum(map) != t_grid.shape[0]:
        idx_filled,  = np.where(map == 1)
        j = idx_filled[:-1]
        k = idx_filled[1:]
        idx_not_in_a_row = np.concatenate([np.diff(j), [2]]) != 1
        j = j[idx_not_in_a_row]
        k = k[idx_not_in_a_row]
        i = np.floor((j + k) / 2).astype(int)
        map[i] = 1

        N_step = rng.normal(size=(size, dim, i.shape[0]))
        E_step = ((t_grid[k] - t_grid[i]) / (t_grid[k] - t_grid[j]) * W[:, :, j] +
                  (t_grid[i] - t_grid[j]) / (t_grid[k] - t_grid[j]) * W[:, :, k])
        std_step = ((t_grid[i] - t_grid[j]) *
                    (t_grid[k] - t_grid[i]) /
                    (t_grid[k] - t_grid[j]))
        W[:, :, i] = E_step + np.sqrt(std_step) * N_step
    if dim == 1:
        return W[:, 0, :]
    return W
