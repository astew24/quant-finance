import unittest
import numpy as np
import pandas as pd

from crypto_volatility.src.risk_utils import calculate_var, calculate_cvar, detect_regimes


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

    def test_cvar_worse_than_var(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.01, 0.03, 1000)
        var = calculate_var(returns, 0.05)
        cvar = calculate_cvar(returns, 0.05)
        self.assertLessEqual(cvar, var, "CVaR should be <= VaR (further into tail)")

    def test_cvar_empty(self):
        with self.assertRaises(ValueError):
            calculate_cvar([], 0.05)

    def test_cvar_invalid_confidence(self):
        with self.assertRaises(ValueError):
            calculate_cvar([0.01], confidence_level=0)

    def test_detect_regimes_labels(self):
        vol = pd.Series([0.1, 0.12, 0.15, 0.5, 0.6, 0.7, 0.11, 0.13, 0.55, 0.65,
                         0.1, 0.12])
        labels, threshold = detect_regimes(vol)
        self.assertEqual(len(labels), len(vol))
        self.assertTrue(set(labels.unique()).issubset({0, 1}))
        self.assertGreater(threshold, 0)

    def test_detect_regimes_too_few(self):
        with self.assertRaises(ValueError):
            detect_regimes(pd.Series([0.1, 0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()
