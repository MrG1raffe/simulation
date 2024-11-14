import unittest
import numpy as np
from scipy.stats import norm

from simulation.monte_carlo import MonteCarlo


class TestMonteCarlo(unittest.TestCase):
    def test_monte_carlo_confidence_level(self):
        n = 101
        m = 20
        sigma = 10
        a = np.ones(n)
        a[::2] *= -1
        a *= sigma
        a += m

        cl = 0.97
        mc = MonteCarlo(batch=a, confidence_level=cl)

        self.assertTrue(np.isclose(mc.mean, -sigma / n + m))
        self.assertTrue(np.isclose(mc.var, 99.99019703950593))
        self.assertTrue(np.isclose(mc.accuracy, 2.159214790545278))
        self.assertTrue(np.isclose(mc.confidence_level, cl))
        self.assertTrue(np.isclose(2 * norm.cdf(- mc.accuracy * np.sqrt(n) / sigma),
                                   1 - mc.confidence_level, rtol=0.001))

        a[0] = 30
        mc.add_batch(a)
        self.assertTrue(np.isclose(mc.mean, m))
        self.assertTrue(np.isclose(mc.var, sigma**2))
        self.assertTrue(np.isclose(mc.accuracy, 2.159214790545278 / np.sqrt(2), rtol=3))
        self.assertTrue(np.isclose(mc.confidence_level, cl))
        self.assertTrue(np.isclose(2 * norm.cdf(- mc.accuracy * np.sqrt(2 * n) / sigma),
                                   1 - mc.confidence_level, rtol=0.001))

        mc.add_batch(a)
        self.assertTrue(np.isclose(mc.accuracy, 2.159214790545278 / np.sqrt(3), rtol=3))
        self.assertTrue(np.isclose(mc.confidence_level, cl))
        self.assertTrue(np.isclose(2 * norm.cdf(- mc.accuracy * np.sqrt(3 * n) / sigma),
                                   1 - mc.confidence_level, rtol=0.001))

    def test_monte_carlo_accuracy(self):
        n = 101
        m = 20
        sigma = 10
        a = np.ones(n)
        a[::2] *= -1
        a *= sigma
        a += m

        eps = 0.5
        mc = MonteCarlo(batch=a, accuracy=eps)

        self.assertTrue(np.isclose(mc.mean, -sigma / n + m))
        self.assertTrue(np.isclose(mc.var, 99.99019703950593))
        self.assertTrue(np.isclose(mc.confidence_level, 0.38469709611374037))
        self.assertTrue(np.isclose(mc.accuracy, eps))
        self.assertTrue(np.isclose(2 * norm.cdf(- mc.accuracy * np.sqrt(n) / sigma),
                                   1 - mc.confidence_level, rtol=0.001))

        a[0] = 30
        mc.add_batch(a)
        self.assertTrue(np.isclose(mc.mean, m))
        self.assertTrue(np.isclose(mc.var, sigma**2))
        self.assertTrue(np.isclose(mc.accuracy, eps))
        self.assertTrue(np.isclose(mc.confidence_level, 0.5226886343465948))
        self.assertTrue(np.isclose(2 * norm.cdf(- mc.accuracy * np.sqrt(2 * n) / sigma),
                                   1 - mc.confidence_level, rtol=0.001))

        mc.add_batch(a)
        self.assertTrue(np.isclose(mc.confidence_level, 0.6158905642232597))
        self.assertTrue(np.isclose(mc.accuracy, eps))
        self.assertTrue(np.isclose(2 * norm.cdf(- mc.accuracy * np.sqrt(3 * n) / sigma),
                                   1 - mc.confidence_level, rtol=0.001))


if __name__ == '__main__':
    unittest.main()