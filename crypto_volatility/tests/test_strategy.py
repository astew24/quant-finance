import unittest

import numpy as np
import pandas as pd

from crypto_volatility.src.strategy import run_volatility_target_strategy


class TestVolatilityTargetStrategy(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        self.returns = pd.Series(
            np.linspace(-0.01, 0.02, len(dates)),
            index=dates,
        )
        self.forecast_vol = pd.Series(0.03, index=dates)

    def test_strategy_runs(self):
        result = run_volatility_target_strategy(
            self.returns,
            self.forecast_vol,
            target_annual_vol=0.30,
            max_leverage=1.5,
            transaction_cost_bps=5.0,
        )
        self.assertEqual(len(result.net_returns), len(self.returns))
        self.assertTrue((result.positions <= 1.5).all())
        self.assertIn("sharpe_ratio", result.summary)
        self.assertIn("average_leverage", result.summary)

    def test_transaction_costs_reduce_returns(self):
        low_cost = run_volatility_target_strategy(
            self.returns,
            self.forecast_vol,
            transaction_cost_bps=0.0,
        )
        high_cost = run_volatility_target_strategy(
            self.returns,
            self.forecast_vol + np.linspace(0.0, 0.02, len(self.forecast_vol)),
            transaction_cost_bps=50.0,
        )
        self.assertLess(
            high_cost.net_returns.sum(),
            low_cost.net_returns.sum(),
        )

    def test_requires_overlap(self):
        with self.assertRaises(ValueError):
            run_volatility_target_strategy(
                self.returns.iloc[:5],
                self.forecast_vol.iloc[:5],
            )


if __name__ == "__main__":
    unittest.main()
