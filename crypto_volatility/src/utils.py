# Additional utility functions for data preprocessing
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    return df.dropna()
