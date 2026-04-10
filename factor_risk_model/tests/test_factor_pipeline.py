import unittest

import numpy as np
import pandas as pd

from factor_risk_model.src.backtesting import run_factor_backtest
from factor_risk_model.src.factor_construction import build_factor_signals


def _make_price_panel(periods: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    base_trends = np.array([0.0012, 0.0010, 0.0008, 0.0004, 0.0002, -0.0001])
    prices = {}
    for i, symbol in enumerate(symbols):
        noise = rng.normal(base_trends[i], 0.01, size=periods)
        prices[symbol] = 100 * np.cumprod(1 + noise)
    return pd.DataFrame(prices, index=dates)


class TestFactorPipeline(unittest.TestCase):
    def setUp(self):
        self.prices = _make_price_panel()
        self.returns = self.prices.pct_change().dropna()

    def test_build_factor_signals(self):
        factor_set = build_factor_signals(self.prices, self.returns)
        self.assertIn("momentum_12_1", factor_set.raw_signals)
        self.assertEqual(
            factor_set.composite_score.shape,
            self.prices.shape,
        )

    def test_backtest_runs(self):
        factor_set = build_factor_signals(self.prices, self.returns)
        result = run_factor_backtest(
            returns=self.returns,
            composite_score=factor_set.composite_score,
            standardized_signals=factor_set.standardized_signals,
            rebalance_frequency=21,
            selection_quantile=1 / 3,
            transaction_cost_bps=10.0,
        )
        self.assertGreater(len(result.portfolio_returns), 0)
        self.assertIn("sharpe_ratio", result.summary)
        self.assertFalse(result.weights.empty)
        self.assertFalse(result.factor_returns.empty)


if __name__ == "__main__":
    unittest.main()
