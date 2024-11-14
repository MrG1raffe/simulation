"""
Three important things to make `python -m unittest` work:
    1. Run it from the root directory "SmileHJM\"
    2. Add an empty file `__init__.py` into each directory with test files.
    3. Test file names should match the pattern "test_*.py". Ideally, use *module_name* after "test_".
"""
import unittest


class TestModule(unittest.TestCase):
    def test_method(self):
        self.assertEqual(2 + 2, 4)


if __name__ == '__main__':
    unittest.main()
