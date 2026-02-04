import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import factor_regression


class TestFactorRegression(unittest.TestCase):
    def test_fit_no_scaling_recovers_exposures(self):
        np.random.seed(0)
        index = pd.date_range("2020-01-01", periods=200, freq="D")
        factors = pd.DataFrame(
            {
                "Market": np.random.normal(0, 0.01, size=200),
                "Value": np.random.normal(0, 0.02, size=200),
            },
            index=index,
        )
        returns = 2.0 * factors["Market"] - 1.0 * factors["Value"] + 0.0001

        model = factor_regression.FactorRegression(method="ols", scale_factors=False)
        model.fit(returns, factors)

        self.assertAlmostEqual(model.exposures["Market"], 2.0, places=2)
        self.assertAlmostEqual(model.exposures["Value"], -1.0, places=2)
        self.assertGreater(model.r_squared, 0.98)

    def test_risk_contributions_sum_to_portfolio_risk(self):
        exposures = pd.Series([0.5, -0.2, 1.0], index=["A", "B", "C"])
        factor_cov = pd.DataFrame(
            [
                [0.04, 0.01, 0.0],
                [0.01, 0.09, 0.02],
                [0.0, 0.02, 0.16],
            ],
            index=exposures.index,
            columns=exposures.index,
        )

        risk = factor_regression.RiskAttribution()
        total_risk = risk.calculate_portfolio_risk(exposures, factor_cov)
        contrib = risk.calculate_risk_contribution(exposures, factor_cov)

        self.assertAlmostEqual(contrib.sum(), total_risk, places=6)

    def test_rolling_regression_shapes(self):
        np.random.seed(1)
        index = pd.date_range("2020-01-01", periods=300, freq="D")
        factors = pd.DataFrame(
            {
                "Market": np.random.normal(0, 0.01, size=300),
                "Momentum": np.random.normal(0, 0.015, size=300),
            },
            index=index,
        )
        returns = 0.7 * factors["Market"] + 0.3 * factors["Momentum"] + np.random.normal(0, 0.001, size=300)

        rolling = factor_regression.RollingFactorRegression(window_size=100)
        exposures_df, r2_series = rolling.fit_and_analyze(returns, factors)

        self.assertEqual(list(exposures_df.columns), list(factors.columns))
        self.assertEqual(len(exposures_df), len(r2_series))


if __name__ == '__main__':
    unittest.main()
