import unittest

import numpy as np
import pandas as pd

from crypto_volatility.src.data_collector import CryptoDataCollector


class TestCryptoDataCollector(unittest.TestCase):
    def setUp(self):
        self.collector = CryptoDataCollector(symbols=["BTC-USD"])

    def test_calculate_returns(self):
        df = pd.DataFrame({"Close": [100.0, 110.0, 121.0]})
        returns = self.collector.calculate_returns(df)
        expected = np.log(np.array([110.0, 121.0]) / np.array([100.0, 110.0]))
        np.testing.assert_almost_equal(returns.dropna().values, expected)

    def test_calculate_returns_missing_column(self):
        df = pd.DataFrame({"price": [100.0, 110.0]})
        with self.assertRaises(KeyError):
            self.collector.calculate_returns(df)

    def test_calculate_volatility(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        vol = self.collector.calculate_volatility(returns, window=3)
        expected = returns[-3:].std() * np.sqrt(365)
        self.assertAlmostEqual(vol.iloc[-1], expected)

    def test_calculate_volatility_invalid_window(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        with self.assertRaises(ValueError):
            self.collector.calculate_volatility(returns, window=1)

    def test_invalid_symbols(self):
        with self.assertRaises(ValueError):
            CryptoDataCollector(symbols=[""])

    def test_ticker_resolution(self):
        c = CryptoDataCollector(symbols=["BTC/USDT"])
        self.assertEqual(c._resolve_ticker("BTC/USDT"), "BTC-USD")


if __name__ == "__main__":
    unittest.main()
