import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import portfolio_opt


class TestPortfolioOpt(unittest.TestCase):
    def test_calculate_sharpe_ratio(self):
        returns = np.array([0.01, 0.02, 0.015, 0.005])
        sharpe = portfolio_opt.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        self.assertTrue(sharpe > 0)

    def test_calculate_sharpe_ratio_zero_variance(self):
        returns = np.array([0.01, 0.01, 0.01])
        with self.assertRaises(ValueError):
            portfolio_opt.calculate_sharpe_ratio(returns)


if __name__ == '__main__':
    unittest.main()
