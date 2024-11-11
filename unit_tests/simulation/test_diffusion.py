import unittest
import numpy as np

from simulation.diffusion import Diffusion


class TestDiffusion(unittest.TestCase):
    def test_brownian_motion(self):
        T = 2
        size = 100
        t_grid = np.linspace(0, T, 1000)
        rng = np.random.default_rng(seed=42)
        diffusion = Diffusion(t_grid=t_grid, size=size, dim=5, rng=rng)

        R = np.eye(3)
        R[0, 1] = R[1, 0] = -0.99999999
        W0 = np.array((1, 2, 3))
        mu = np.array((2, 2, 5))
        sigma = np.array((2, 1, 2))
        W = diffusion.brownian_motion(init_val=W0, drift=mu, correlation=R, vol=sigma, dims=(1, 2, 3))

        self.assertTrue(W.shape == (size, 3, len(t_grid)))
        self.assertTrue(np.allclose((W[:, 0, :] - W0[0] - mu[0] * t_grid) / sigma[0],
                                    -(W[:, 1, :] - W0[1] - mu[1] * t_grid) / sigma[1],
                                    atol=0.001)
        )
        self.assertTrue(np.allclose(np.corrcoef(np.diff(W[0, 1, :]), np.diff(W[0, 2, :])), np.eye(2), atol=0.03))

        S = diffusion.geometric_brownian_motion(init_val=np.exp(W0), drift=mu, correlation=R, vol=sigma, dims=(1, 2, 3))
        W = np.log(S)
        self.assertTrue(S.shape == (size, 3, len(t_grid)))
        self.assertTrue(np.allclose((W[:, 0, :] - W0[0] - (mu[0] - 0.5 * sigma[0]**2) * t_grid) / sigma[0],
                                    -(W[:, 1, :] - W0[1] - (mu[1] - 0.5 * sigma[1]**2) * t_grid) / sigma[1],
                                    atol=0.001)
                        )
        self.assertTrue(np.allclose(np.corrcoef(np.diff(W[0, 1, :]), np.diff(W[0, 2, :])), np.eye(2), atol=0.03))

    def test_brownian_motion_moments(self):
        T = 2
        size = 10_000
        t_grid = np.linspace(0, T, 2)
        rng = np.random.default_rng(seed=42)
        diffusion = Diffusion(t_grid=t_grid, size=size, dim=5, rng=rng)

        R = np.eye(3)
        R[0, 1] = R[1, 0] = -0.99999999
        W0 = np.array((3, 2, 3))
        mu = np.array((-4, 4, 5))
        sigma = np.array((5, 1, 2))
        W = diffusion.brownian_motion(init_val=W0, drift=mu, correlation=R, vol=sigma, dims=(0, 2, 4))

        self.assertTrue(W.shape == (size, 3, len(t_grid)))
        self.assertTrue(np.allclose(W[:, :, 1].mean(0), W0 + T * np.array(mu), rtol=0.05))
        self.assertTrue(np.allclose(W[:, :, 1].var(0), np.array(sigma) ** 2 * T, rtol=0.5))

    def test_brownian_motion_rng(self):
        T = 2
        size = 1000
        t_grid = np.linspace(0, T, 2)
        rng1 = np.random.default_rng(seed=42)
        diffusion1 = Diffusion(t_grid=t_grid, size=size, dim=5, rng=rng1)
        rng2 = np.random.default_rng(seed=42)
        diffusion2 = Diffusion(t_grid=t_grid, size=size, dim=5, rng=rng2)

        R = np.eye(3)
        R[0, 1] = R[1, 0] = -0.99999999
        W0 = np.array((3, 2, 3))
        mu = np.array((-4, 4, 5))
        sigma = np.array((5, 1, 2))
        W1 = diffusion1.brownian_motion(init_val=W0, drift=mu, correlation=R, vol=sigma, dims=(0, 2, 4))
        W2 = diffusion2.brownian_motion(init_val=W0, drift=mu, correlation=R, vol=sigma, dims=(0, 2, 4))

        self.assertTrue(np.allclose(W1, W2))


if __name__ == '__main__':
    unittest.main()
