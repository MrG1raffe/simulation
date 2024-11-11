import unittest
import numpy as np

from simulation.utility import *


class TestUtility(unittest.TestCase):
    def test_is_number(self):
        self.assertTrue(is_number(2))
        self.assertTrue(is_number(5.8))
        self.assertTrue(is_number(np.pi))
        self.assertTrue(is_number(0))
        self.assertTrue(is_number(1 + 1j * 3.3))
        self.assertTrue(is_number(1j))
        self.assertFalse(is_number([1, 2, 3]))
        self.assertFalse(is_number((1, 2.4, "R")))
        self.assertFalse(is_number("abcdef"))
        self.assertFalse(is_number(np.ones(5)))

    def test_to_numpy(self):
        self.assertTrue(isinstance(to_numpy(1), np.ndarray))
        self.assertTrue(isinstance(to_numpy([1, 2.5]), np.ndarray))
        self.assertTrue(to_numpy(2.5).shape == (1,))
        self.assertTrue(to_numpy(np.ones(10)).shape == np.ones(10).shape)
        self.assertTrue(to_numpy(np.ones((1, 2, 3)).shape == np.ones((1, 2, 3)).shape))


if __name__ == '__main__':
    unittest.main()