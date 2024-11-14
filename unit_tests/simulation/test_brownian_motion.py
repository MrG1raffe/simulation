import unittest
import numpy as np

from simulation.brownian_motion import simulate_brownian_motion_from_increments


class TestSimulation(unittest.TestCase):
    def test_simulate_brownian_motion_from_increments(self):
        rng = np.random.default_rng(seed=42)
        t_grid = np.linspace(0, 2, 1000)
        size = 1_000
        dim = 3
        B = simulate_brownian_motion_from_increments(size=size, t_grid=t_grid, rng=rng, dim=dim)

        self.assertTrue(B.shape == (size, dim, len(t_grid)))
        self.assertTrue(np.allclose(np.mean(B, 0), 0, atol=1e-1))
        self.assertTrue(np.allclose(np.var(B, 0), t_grid, rtol=1e-0))
        self.assertTrue(np.isclose(np.var(np.diff(B, axis=2)), t_grid[1] - t_grid[0], rtol=1e-3))
        self.assertTrue(np.isclose(np.mean(np.diff(B, axis=2)), 0, atol=1e-3))
        self.assertTrue(np.allclose(np.corrcoef(np.diff(B, axis=2)[0]), np.eye(dim), atol=0.05))


if __name__ == '__main__':
    unittest.main()
