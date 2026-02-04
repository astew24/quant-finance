import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import metrics


class TestMetrics(unittest.TestCase):
    def test_calculate_rmse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 2.0])
        rmse = metrics.calculate_rmse(y_true, y_pred)
        self.assertAlmostEqual(rmse, np.sqrt(((0.0)**2 + (0.5)**2 + (-1.0)**2) / 3), places=6)


if __name__ == '__main__':
    unittest.main()
