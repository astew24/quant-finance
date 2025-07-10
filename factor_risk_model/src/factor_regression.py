"""
Factor Regression Module

This module implements factor regression analysis for risk modeling.
Provides functionality for OLS and Ridge regression, factor exposure analysis,
and risk attribution calculations.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FactorRegression:
    """
    Factor regression analysis for risk modeling.
    
    Attributes:
        model: Fitted regression model
        exposures: Factor exposures
        r_squared: R-squared value
        residuals: Model residuals
    """
    
    def __init__(self, method: str = 'ols', alpha: float = 1.0):
        """
        Initialize factor regression.
        
        Args:
            method: Regression method ('ols' or 'ridge')
            alpha: Regularization parameter for Ridge regression
        """
        self.method = method
        self.alpha = alpha
        self.model = None
        self.exposures = None
        self.r_squared = None
        self.residuals = None
        self.scaler = StandardScaler()
        
    def fit(self, returns: pd.Series, factors: pd.DataFrame) -> 'FactorRegression':
        """
        Fit factor regression model.
        
        Args:
            returns: Stock returns series
            factors: Factor returns DataFrame
            
        Returns:
            Self for method chaining
        """
        try:
            # Align data
            common_index = returns.index.intersection(factors.index)
            returns_aligned = returns.loc[common_index]
            factors_aligned = factors.loc[common_index]
            
            # Remove NaN values
            valid_mask = ~(returns_aligned.isna() | factors_aligned.isna().any(axis=1))
            returns_clean = returns_aligned[valid_mask]
            factors_clean = factors_aligned[valid_mask]
            
            if len(returns_clean) == 0:
                raise ValueError("No valid data points after removing NaN values")
            
            # Scale factors
            factors_scaled = self.scaler.fit_transform(factors_clean)
            
            # Fit model
            if self.method == 'ols':
                self.model = LinearRegression()
            elif self.method == 'ridge':
                self.model = Ridge(alpha=self.alpha)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.model.fit(factors_scaled, returns_clean)
            
            # Calculate exposures
            self.exposures = pd.Series(
                self.model.coef_,
                index=factors_clean.columns
            )
            
            # Calculate R-squared
            predictions = self.model.predict(factors_scaled)
            self.r_squared = r2_score(returns_clean, predictions)
            
            # Calculate residuals
            self.residuals = returns_clean - predictions
            
            logger.info(f"Factor regression fitted successfully")
            logger.info(f"R-squared: {self.r_squared:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting factor regression: {str(e)}")
            raise
            
        return self
    
    def predict(self, factors: pd.DataFrame) -> pd.Series:
        """
        Generate predictions using fitted model.
        
        Args:
            factors: Factor returns DataFrame
            
        Returns:
            Series of predicted returns
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
            
        try:
            # Scale factors
            factors_scaled = self.scaler.transform(factors)
            
            # Generate predictions
            predictions = self.model.predict(factors_scaled)
            
            return pd.Series(predictions, index=factors.index)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def get_model_summary(self) -> Dict:
        """
        Get model summary statistics.
        
        Returns:
            Dictionary with model summary
        """
        if self.model is None:
            return {}
            
        return {
            'method': self.method,
            'r_squared': self.r_squared,
            'exposures': self.exposures.to_dict(),
            'residuals_mean': self.residuals.mean(),
            'residuals_std': self.residuals.std(),
            'alpha': self.alpha if self.method == 'ridge' else None
        }


class RollingFactorRegression:
    """
    Rolling window factor regression for time-varying factor exposures.
    """
    
    def __init__(self, window_size: int = 252, method: str = 'ols', alpha: float = 1.0):
        """
        Initialize rolling factor regression.
        
        Args:
            window_size: Size of rolling window
            method: Regression method
            alpha: Regularization parameter
        """
        self.window_size = window_size
        self.method = method
        self.alpha = alpha
        self.exposures_history = []
        self.r_squared_history = []
        
    def fit_and_analyze(self, returns: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Fit rolling factor regression and analyze time-varying exposures.
        
        Args:
            returns: Stock returns series
            factors: Factor returns DataFrame
            
        Returns:
            DataFrame with time-varying factor exposures
        """
        # Align data
        common_index = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_index]
        factors_aligned = factors.loc[common_index]
        
        exposures_df = pd.DataFrame(index=returns_aligned.index[self.window_size:])
        r_squared_series = pd.Series(index=returns_aligned.index[self.window_size:])
        
        for i in range(self.window_size, len(returns_aligned)):
            # Training window
            train_returns = returns_aligned.iloc[i-self.window_size:i]
            train_factors = factors_aligned.iloc[i-self.window_size:i]
            
            # Remove NaN values
            valid_mask = ~(train_returns.isna() | train_factors.isna().any(axis=1))
            train_returns_clean = train_returns[valid_mask]
            train_factors_clean = train_factors[valid_mask]
            
            if len(train_returns_clean) < self.window_size * 0.8:  # Require 80% data
                exposures_df.iloc[i-self.window_size] = np.nan
                r_squared_series.iloc[i-self.window_size] = np.nan
                continue
            
            try:
                # Fit regression
                regression = FactorRegression(method=self.method, alpha=self.alpha)
                regression.fit(train_returns_clean, train_factors_clean)
                
                # Store results
                exposures_df.iloc[i-self.window_size] = regression.exposures
                r_squared_series.iloc[i-self.window_size] = regression.r_squared
                
            except Exception as e:
                logger.warning(f"Error in rolling window {i}: {str(e)}")
                exposures_df.iloc[i-self.window_size] = np.nan
                r_squared_series.iloc[i-self.window_size] = np.nan
        
        return exposures_df, r_squared_series


class RiskAttribution:
    """
    Risk attribution analysis using factor models.
    """
    
    def __init__(self):
        """
        Initialize risk attribution calculator.
        """
        pass
    
    def calculate_factor_covariance(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate factor covariance matrix.
        
        Args:
            factors: Factor returns DataFrame
            
        Returns:
            Factor covariance matrix
        """
        return factors.cov()
    
    def calculate_portfolio_risk(self, exposures: pd.Series, 
                               factor_cov: pd.DataFrame) -> float:
        """
        Calculate portfolio risk using factor model.
        
        Args:
            exposures: Factor exposures
            factor_cov: Factor covariance matrix
            
        Returns:
            Portfolio risk (volatility)
        """
        # Portfolio risk = sqrt(exposures' * factor_cov * exposures)
        risk = np.sqrt(exposures.T @ factor_cov @ exposures)
        return risk
    
    def calculate_risk_contribution(self, exposures: pd.Series,
                                  factor_cov: pd.DataFrame) -> pd.Series:
        """
        Calculate risk contribution of each factor.
        
        Args:
            exposures: Factor exposures
            factor_cov: Factor covariance matrix
            
        Returns:
            Series with risk contribution of each factor
        """
        portfolio_risk = self.calculate_portfolio_risk(exposures, factor_cov)
        
        # Risk contribution = (factor_cov * exposures) / portfolio_risk
        risk_contrib = (factor_cov @ exposures) / portfolio_risk
        
        return pd.Series(risk_contrib, index=exposures.index)
    
    def calculate_marginal_risk(self, exposures: pd.Series,
                               factor_cov: pd.DataFrame) -> pd.Series:
        """
        Calculate marginal risk contribution of each factor.
        
        Args:
            exposures: Factor exposures
            factor_cov: Factor covariance matrix
            
        Returns:
            Series with marginal risk contribution of each factor
        """
        portfolio_risk = self.calculate_portfolio_risk(exposures, factor_cov)
        
        # Marginal risk contribution = (factor_cov * exposures) / portfolio_risk
        marginal_risk = (factor_cov @ exposures) / portfolio_risk
        
        return pd.Series(marginal_risk, index=exposures.index)


def analyze_multiple_stocks(returns_data: pd.DataFrame, factors: pd.DataFrame,
                          method: str = 'ols', alpha: float = 1.0) -> Dict[str, Dict]:
    """
    Analyze factor exposures for multiple stocks.
    
    Args:
        returns_data: DataFrame with stock returns
        factors: Factor returns DataFrame
        method: Regression method
        alpha: Regularization parameter
        
    Returns:
        Dictionary with analysis results for each stock
    """
    results = {}
    
    for symbol in returns_data.columns:
        try:
            returns = returns_data[symbol].dropna()
            
            if len(returns) < 100:  # Require minimum data points
                logger.warning(f"Insufficient data for {symbol}")
                continue
            
            # Fit factor regression
            regression = FactorRegression(method=method, alpha=alpha)
            regression.fit(returns, factors)
            
            results[symbol] = regression.get_model_summary()
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            results[symbol] = {'error': str(e)}
    
    return results


def calculate_portfolio_risk_attribution(portfolio_weights: pd.Series,
                                       stock_exposures: pd.DataFrame,
                                       factor_cov: pd.DataFrame) -> Dict:
    """
    Calculate portfolio-level risk attribution.
    
    Args:
        portfolio_weights: Portfolio weights for each stock
        stock_exposures: Factor exposures for each stock
        factor_cov: Factor covariance matrix
        
    Returns:
        Dictionary with portfolio risk attribution
    """
    # Calculate portfolio factor exposures
    portfolio_exposures = (stock_exposures.T * portfolio_weights).sum(axis=1)
    
    # Calculate portfolio risk
    risk_attribution = RiskAttribution()
    portfolio_risk = risk_attribution.calculate_portfolio_risk(portfolio_exposures, factor_cov)
    risk_contrib = risk_attribution.calculate_risk_contribution(portfolio_exposures, factor_cov)
    
    return {
        'portfolio_risk': portfolio_risk,
        'portfolio_exposures': portfolio_exposures,
        'risk_contribution': risk_contrib
    }


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Get sample data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    returns_data = pd.DataFrame()
    
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(start='2020-01-01', end='2023-01-01')
        returns = np.log(data['Close'] / data['Close'].shift(1))
        returns_data[symbol] = returns
    
    # Create sample factors
    factors = pd.DataFrame({
        'Market': returns_data.mean(axis=1),
        'Momentum': returns_data.rolling(30).mean().mean(axis=1)
    })
    
    # Analyze factor exposures
    results = analyze_multiple_stocks(returns_data, factors)
    
    print("Factor analysis complete!")
    for symbol, result in results.items():
        if 'error' not in result:
            print(f"{symbol}: R² = {result['r_squared']:.4f}") 