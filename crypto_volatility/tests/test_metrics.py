import unittest

import numpy as np

from crypto_volatility.src.metrics import calculate_rmse


class TestMetrics(unittest.TestCase):
    def test_calculate_rmse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 2.0])
        rmse = calculate_rmse(y_true, y_pred)
        expected = np.sqrt(((0.0) ** 2 + (0.5) ** 2 + (-1.0) ** 2) / 3)
        self.assertAlmostEqual(rmse, expected, places=6)

    def test_empty_inputs(self):
        with self.assertRaises(ValueError):
            calculate_rmse([], [])

    def test_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            calculate_rmse([1, 2], [1])


if __name__ == "__main__":
    unittest.main()
