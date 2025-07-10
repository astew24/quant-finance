# Risk management utilities
import numpy as np
def calculate_var(returns, confidence_level=0.05):
    return np.percentile(returns, confidence_level * 100)
