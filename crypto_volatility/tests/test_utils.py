import unittest

import numpy as np
import pandas as pd

from crypto_volatility.src.utils import clean_data


class TestUtils(unittest.TestCase):
    def test_clean_data_removes_nans(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
        cleaned = clean_data(df)
        self.assertFalse(cleaned.isnull().values.any())
        self.assertEqual(len(cleaned), 1)

    def test_clean_data_type_error(self):
        with self.assertRaises(TypeError):
            clean_data([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
