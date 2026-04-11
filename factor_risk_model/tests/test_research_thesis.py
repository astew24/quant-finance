import unittest

import numpy as np
import pandas as pd

from factor_risk_model.src.research_thesis import (
    build_quantamental_theses,
    estimate_cost_of_equity,
    simple_dcf_per_share,
)


def _make_screen() -> pd.DataFrame:
    index = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    return pd.DataFrame(
        {
            "name": index,
            "sector": ["Tech", "Tech", "Finance", "Energy", "Healthcare"],
            "forward_pe": [14.0, 18.0, 12.0, 10.0, 20.0],
            "price_to_book": [2.0, 3.0, 1.2, 1.8, 4.5],
            "value_score": [0.9, 0.2, 0.5, 0.6, -0.2],
            "momentum_score": [0.8, 0.4, 0.3, 0.7, 0.1],
            "quality_score": [0.7, 0.6, 0.2, 0.4, 0.8],
            "factor_score": [0.82, 0.42, 0.34, 0.58, 0.19],
            "screen_score": [0.80, 0.45, 0.35, 0.57, 0.18],
            "screen_rank": [1, 2, 4, 3, 5],
            "predicted_outperformance_probability": [0.58, 0.55, 0.49, 0.53, 0.47],
            "free_cashflow": [5e9, 4e9, 3e9, 6e9, 2e9],
            "shares_outstanding": [1e9, 1.2e9, 0.8e9, 0.9e9, 1.1e9],
            "total_cash": [2e9, 1e9, 1e9, 1.5e9, 0.5e9],
            "total_debt": [1e9, 2e9, 3e9, 2e9, 1e9],
            "earnings_growth": [0.08, 0.05, 0.03, 0.06, 0.04],
            "beta": [1.1, 1.0, 0.9, 1.2, 0.8],
            "currentPrice": [120.0, 95.0, 60.0, 110.0, 85.0],
            "targetMeanPrice": [138.0, 102.0, 63.0, 119.0, 90.0],
            "debt_to_equity": [40.0, 80.0, 120.0, 60.0, 50.0],
        },
        index=index,
    )


def _make_prices() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    index = pd.date_range("2024-01-01", periods=300, freq="B")
    data = {}
    for symbol, drift in zip(["AAA", "BBB", "CCC", "DDD", "EEE"], [0.001, 0.0008, 0.0002, 0.0006, 0.0004]):
        data[symbol] = 100 * np.cumprod(1 + rng.normal(drift, 0.015, size=len(index)))
    return pd.DataFrame(data, index=index)


class TestResearchThesis(unittest.TestCase):
    def test_estimate_cost_of_equity(self):
        value = estimate_cost_of_equity(1.2)
        self.assertGreater(value, 0.04)

    def test_simple_dcf(self):
        fair_value = simple_dcf_per_share(
            free_cash_flow=5e9,
            shares_outstanding=1e9,
            total_cash=2e9,
            total_debt=1e9,
            growth_rate=0.08,
            cost_of_equity=0.10,
        )
        self.assertIsNotNone(fair_value)
        self.assertGreater(fair_value, 0)

    def test_build_quantamental_theses(self):
        result = build_quantamental_theses(_make_screen(), _make_prices(), top_n=3)
        self.assertEqual(len(result.ideas_table), 3)
        self.assertIn("DCF", result.markdown_report)
        self.assertIn("street target implies", result.markdown_report)
        self.assertIn("AAA", result.markdown_report)


if __name__ == "__main__":
    unittest.main()
