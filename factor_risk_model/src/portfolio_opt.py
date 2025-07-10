# Portfolio optimization utilities
import numpy as np
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    return excess_returns.mean() / excess_returns.std()
