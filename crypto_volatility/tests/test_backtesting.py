import unittest
import numpy as np
import pandas as pd

from crypto_volatility.src.backtesting import (
    diebold_mariano_test,
    naive_baselines,
    walk_forward_garch,
)


def _make_returns(n=400, seed=42):
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.05, 0.10, 0.85
    sigma2 = np.zeros(n)
    rets = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * rets[t - 1] ** 2 + beta * sigma2[t - 1]
        rets[t] = rng.normal(0, np.sqrt(sigma2[t]))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(rets, index=dates)


class TestDieboldMariano(unittest.TestCase):
    def test_equal_errors(self):
        e = np.random.randn(100)
        stat, pval = diebold_mariano_test(e, e)
        # identical errors -> stat ~0, pval ~1
        if not np.isnan(stat):
            self.assertGreater(pval, 0.5)

    def test_clearly_different(self):
        e1 = np.ones(200) * 0.01
        e2 = np.ones(200) * 10.0
        stat, pval = diebold_mariano_test(e1, e2)
        self.assertLess(pval, 0.05, "Should detect significant difference")

    def test_too_few(self):
        stat, pval = diebold_mariano_test(np.array([1.0]), np.array([2.0]))
        self.assertTrue(np.isnan(stat))


class TestNaiveBaselines(unittest.TestCase):
    def test_returns_dict(self):
        rets = _make_returns(500)
        baselines = naive_baselines(rets, window=100, step=10)
        self.assertIsInstance(baselines, dict)
        self.assertGreater(len(baselines), 0)
        for name, res in baselines.items():
            self.assertGreater(res.n_windows, 0)
            self.assertGreater(res.rmse, 0)


class TestWalkForwardGarch(unittest.TestCase):
    def test_runs_and_returns_result(self):
        rets = _make_returns(400)
        result = walk_forward_garch(rets, window=100, step=20)
        self.assertEqual(result.name, "GARCH(1,1)")
        self.assertGreater(result.n_windows, 0)
        self.assertGreater(result.direction_accuracy, 0)
        self.assertGreater(result.rmse, 0)

    def test_too_short(self):
        rets = _make_returns(50)
        with self.assertRaises(ValueError):
            walk_forward_garch(rets, window=100)


if __name__ == "__main__":
    unittest.main()
