import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import utils

class TestUtils(unittest.TestCase):
    def test_dummy(self):
        # Replace with a real test for utils.py
        self.assertTrue(True)

    def test_clean_data_removes_nans(self):
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
        cleaned = utils.clean_data(df)
        self.assertFalse(cleaned.isnull().values.any())
        self.assertEqual(len(cleaned), 1)

if __name__ == '__main__':
    unittest.main() 