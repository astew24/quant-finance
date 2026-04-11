import unittest

import numpy as np
import pandas as pd

from factor_risk_model.src.screening import (
    FEATURE_COLUMNS,
    build_classification_dataset,
    build_factor_screen,
    fit_return_classifier,
)


def _make_screen_prices(periods: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2021-01-01", periods=periods, freq="B")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    drifts = np.array([0.0010, 0.0008, 0.0006, 0.0004, 0.0001, -0.0001, -0.0002, -0.0004])
    prices = {}
    for i, symbol in enumerate(symbols):
        innovations = rng.normal(drifts[i], 0.012, size=periods)
        prices[symbol] = 100 * np.cumprod(1 + innovations)
    return pd.DataFrame(prices, index=dates)


def _make_fundamentals(symbols) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shortName": symbols,
            "sector": ["Tech", "Tech", "Tech", "Finance", "Finance", "Energy", "Energy", "Industrial"],
            "trailingPE": [12, 14, 16, 18, 20, 22, 24, 26],
            "forwardPE": [11, 13, 15, 17, 19, 21, 23, 25],
            "priceToBook": [1.1, 1.3, 1.5, 1.8, 2.0, 2.3, 2.6, 2.8],
            "enterpriseToEbitda": [6, 7, 8, 9, 10, 11, 12, 13],
            "returnOnEquity": [0.25, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10],
            "returnOnAssets": [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05],
            "profitMargins": [0.24, 0.22, 0.21, 0.18, 0.17, 0.14, 0.12, 0.10],
            "operatingMargins": [0.28, 0.25, 0.23, 0.20, 0.18, 0.16, 0.13, 0.11],
            "currentRatio": [2.1, 2.0, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2],
            "debtToEquity": [20, 30, 40, 50, 60, 70, 80, 90],
        },
        index=symbols,
    )


class TestScreeningPipeline(unittest.TestCase):
    def setUp(self):
        self.prices = _make_screen_prices()
        self.fundamentals = _make_fundamentals(list(self.prices.columns))

    def test_build_factor_screen(self):
        screen = build_factor_screen(self.prices, self.fundamentals)
        self.assertIn("value_score", screen.columns)
        self.assertIn("quality_score", screen.columns)
        self.assertIn("momentum_score", screen.columns)
        self.assertEqual(int(screen.iloc[0]["factor_rank"]), 1)

    def test_build_classification_dataset(self):
        dataset = build_classification_dataset(self.prices)
        self.assertGreater(len(dataset), 0)
        for column in FEATURE_COLUMNS:
            self.assertIn(column, dataset.columns)
        self.assertIn("target", dataset.columns)

    def test_fit_return_classifier(self):
        dataset = build_classification_dataset(self.prices)
        _, metrics, coefficients, holdout = fit_return_classifier(dataset)
        self.assertIn("holdout_accuracy", metrics)
        self.assertIn("holdout_roc_auc", metrics)
        self.assertFalse(coefficients.empty)
        self.assertTrue(
            holdout["predicted_probability"].between(0.0, 1.0).all()
        )


if __name__ == "__main__":
    unittest.main()
