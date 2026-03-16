import unittest

import numpy as np

from crypto_volatility.src.risk_utils import calculate_var


class TestRiskUtils(unittest.TestCase):
    def test_calculate_var(self):
        returns = np.array([-0.05, -0.02, 0.01, 0.02, 0.03])
        var_5 = calculate_var(returns, confidence_level=0.05)
        self.assertAlmostEqual(var_5, np.percentile(returns, 5), places=6)

    def test_var_invalid_confidence(self):
        with self.assertRaises(ValueError):
            calculate_var([0.01, 0.02], confidence_level=1.5)

    def test_var_empty(self):
        with self.assertRaises(ValueError):
            calculate_var([], confidence_level=0.05)


if __name__ == "__main__":
    unittest.main()
