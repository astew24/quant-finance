# Visualization utilities for factor analysis
from typing import Union
import pandas as pd
import plotly.graph_objects as go


def plot_factor_exposures(exposures: Union[pd.DataFrame, pd.Series]) -> go.Figure:
    if not isinstance(exposures, (pd.DataFrame, pd.Series)):
        raise TypeError("exposures must be a pandas DataFrame or Series")
    return go.Figure()
