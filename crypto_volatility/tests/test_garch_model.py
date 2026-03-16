"""Tests for the GARCH model module."""

import unittest

import numpy as np
import pandas as pd

from crypto_volatility.src.garch_model import GARCHModel, compare_garch_models


def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate synthetic return series with volatility clustering."""
    rng = np.random.default_rng(seed)
    # Simple GARCH(1,1)-like DGP
    omega, alpha, beta = 0.05, 0.10, 0.85
    sigma2 = np.zeros(n)
    returns = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(returns, index=dates)


class TestGARCHModel(unittest.TestCase):
    def setUp(self):
        self.returns = _make_returns()

    def test_fit_returns_self(self):
        garch = GARCHModel()
        result = garch.fit(self.returns)
        self.assertIs(result, garch)

    def test_fit_populates_params(self):
        garch = GARCHModel()
        garch.fit(self.returns)
        self.assertIsNotNone(garch.params)
        self.assertIn("omega", garch.params.index)

    def test_forecast_shape(self):
        garch = GARCHModel()
        garch.fit(self.returns)
        fc = garch.forecast(horizon=5)
        self.assertEqual(len(fc), 5)
        self.assertTrue((fc > 0).all(), "Forecasted volatility should be positive")

    def test_forecast_without_fit_raises(self):
        garch = GARCHModel()
        with self.assertRaises(ValueError):
            garch.forecast(horizon=5)

    def test_model_summary_keys(self):
        garch = GARCHModel()
        garch.fit(self.returns)
        summary = garch.get_model_summary()
        for key in ("aic", "bic", "log_likelihood", "params", "conditional_volatility"):
            self.assertIn(key, summary)

    def test_evaluate_forecasts(self):
        dates = pd.date_range("2023-01-01", periods=10)
        actual = pd.Series(np.random.rand(10), index=dates)
        forecast = pd.Series(np.random.rand(10), index=dates)
        garch = GARCHModel()
        metrics = garch.evaluate_forecasts(actual, forecast)
        for key in ("mse", "mae", "rmse", "direction_accuracy"):
            self.assertIn(key, metrics)
        self.assertGreaterEqual(metrics["rmse"], 0)

    def test_compare_garch_models(self):
        specs = {"GARCH(1,1)": (1, 1), "GARCH(1,2)": (1, 2)}
        results = compare_garch_models(self.returns, specs)
        self.assertIn("GARCH(1,1)", results)
        self.assertIn("aic", results["GARCH(1,1)"])


if __name__ == "__main__":
    unittest.main()
