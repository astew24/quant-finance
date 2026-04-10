import unittest

import numpy as np
import pandas as pd

from crypto_volatility.src.performance import (
    annualized_volatility,
    cumulative_returns,
    max_drawdown,
    summarize_strategy,
)


class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.returns = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])

    def test_cumulative_returns(self):
        curve = cumulative_returns(self.returns)
        expected = (1.0 + self.returns).cumprod() - 1.0
        np.testing.assert_allclose(curve.values, expected.values)

    def test_annualized_volatility_positive(self):
        vol = annualized_volatility(self.returns, periods_per_year=252)
        self.assertGreater(vol, 0.0)

    def test_max_drawdown_negative(self):
        drawdown = max_drawdown(self.returns)
        self.assertLessEqual(drawdown, 0.0)

    def test_summarize_strategy_includes_benchmark_fields(self):
        benchmark = pd.Series([0.008, -0.01, 0.01, 0.004, -0.006])
        turnover = pd.Series([0.5, 0.1, 0.2, 0.0, 0.3])
        summary = summarize_strategy(
            self.returns,
            benchmark_returns=benchmark,
            turnover=turnover,
            periods_per_year=252,
        )
        self.assertIn("sharpe_ratio", summary)
        self.assertIn("information_ratio", summary)
        self.assertIn("average_turnover", summary)


if __name__ == "__main__":
    unittest.main()
