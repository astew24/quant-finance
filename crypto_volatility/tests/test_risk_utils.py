import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import risk_utils


class TestRiskUtils(unittest.TestCase):
    def test_calculate_var(self):
        returns = np.array([-0.05, -0.02, 0.01, 0.02, 0.03])
        var_5 = risk_utils.calculate_var(returns, confidence_level=0.05)
        self.assertAlmostEqual(var_5, np.percentile(returns, 5), places=6)


if __name__ == '__main__':
    unittest.main()
