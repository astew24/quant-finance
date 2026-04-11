import unittest

import pandas as pd

from crypto_volatility.src.demo_data import (
    available_sample_symbols,
    build_normalized_price_index,
    load_sample_market_data,
    sample_data_exists,
)


class TestDemoData(unittest.TestCase):
    def test_available_sample_symbols(self):
        symbols = available_sample_symbols()
        self.assertIn("BTC-USD", symbols)
        self.assertIn("ETH-USD", symbols)

    def test_sample_data_exists(self):
        self.assertTrue(sample_data_exists("BTC-USD"))
        self.assertFalse(sample_data_exists("DOGE-USD"))

    def test_build_normalized_price_index(self):
        returns = pd.Series([0.0, 0.1, -0.1])
        prices = build_normalized_price_index(returns, base_price=100.0)
        self.assertEqual(prices.iloc[0], 100.0)
        self.assertEqual(prices.name, "Close")

    def test_load_sample_market_data(self):
        sample = load_sample_market_data("BTC-USD", days=30)
        self.assertEqual(len(sample), 30)
        self.assertIn("Close", sample.columns)
        self.assertIn("returns", sample.columns)
        self.assertTrue(sample.index.is_monotonic_increasing)
        self.assertGreater(sample["Close"].iloc[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
