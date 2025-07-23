import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pandas as pd
import numpy as np
from data_collector import CryptoDataCollector

class TestCryptoDataCollector(unittest.TestCase):
    def setUp(self):
        self.collector = CryptoDataCollector()

    def test_calculate_returns(self):
        df = pd.DataFrame({'close': [100, 110, 121]})
        returns = self.collector.calculate_returns(df)
        expected = np.log(np.array([110, 121]) / np.array([100, 110]))
        np.testing.assert_almost_equal(returns.dropna().values, expected)

    def test_calculate_volatility(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        vol = self.collector.calculate_volatility(returns, window=3)
        # Check rolling std * sqrt(252) for last value
        expected = returns[-3:].std() * np.sqrt(252)
        self.assertAlmostEqual(vol.iloc[-1], expected)

if __name__ == '__main__':
    unittest.main()