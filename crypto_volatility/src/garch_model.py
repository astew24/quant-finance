"""
GARCH Model Module

This module implements GARCH(1,1) modeling for volatility forecasting.
Provides functionality for model fitting, forecasting, and evaluation.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GARCHModel:
    """
    GARCH(1,1) model for volatility forecasting.
    
    Attributes:
        model: Fitted GARCH model
        params: Model parameters
        forecasts: Model forecasts
    """
    
    def __init__(self, p: int = 1, q: int = 1, mean: str = 'Zero', vol: str = 'GARCH'):
        """
        Initialize GARCH model.
        
        Args:
            p: Number of GARCH terms
            q: Number of ARCH terms
            mean: Mean model specification
            vol: Volatility model specification
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.model = None
        self.params = None
        self.forecasts = None
        
    def fit(self, returns: pd.Series) -> 'GARCHModel':
        """
        Fit GARCH model to return series.
        
        Args:
            returns: Series of log returns
            
        Returns:
            Self for method chaining
        """
        try:
            # Remove NaN values
            returns_clean = returns.dropna()
            
            # Create and fit GARCH model
            self.model = arch_model(
                returns_clean, 
                vol=vol, 
                p=p, 
                q=q, 
                mean=mean,
                dist='normal'
            )
            
            # Fit the model
            self.fitted_model = self.model.fit(disp='off')
            self.params = self.fitted_model.params
            
            logger.info(f"GARCH({p},{q}) model fitted successfully")
            logger.info(f"Model parameters: {self.params}")
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {str(e)}")
            raise
            
        return self
    
    def forecast(self, horizon: int = 1) -> pd.Series:
        """
        Generate volatility forecasts.
        
        Args:
            horizon: Forecast horizon in periods
            
        Returns:
            Series of volatility forecasts
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
            
        try:
            # Generate forecasts
            forecast = self.fitted_model.forecast(horizon=horizon)
            
            # Extract conditional volatility
            self.forecasts = np.sqrt(forecast.variance.values[-1, :])
            
            # Create forecast dates
            last_date = self.fitted_model.data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            return pd.Series(self.forecasts, index=forecast_dates)
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {str(e)}")
            raise
    
    def get_model_summary(self) -> Dict:
        """
        Get model summary statistics.
        
        Returns:
            Dictionary with model summary
        """
        if self.fitted_model is None:
            return {}
            
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.loglikelihood,
            'params': self.params.to_dict(),
            'residuals': self.fitted_model.resid,
            'conditional_volatility': self.fitted_model.conditional_volatility
        }
    
    def evaluate_forecasts(self, actual_vol: pd.Series, 
                          forecast_vol: pd.Series) -> Dict[str, float]:
        """
        Evaluate forecast accuracy.
        
        Args:
            actual_vol: Actual volatility series
            forecast_vol: Forecasted volatility series
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Align series
        common_index = actual_vol.index.intersection(forecast_vol.index)
        actual_aligned = actual_vol.loc[common_index]
        forecast_aligned = forecast_vol.loc[common_index]
        
        # Calculate metrics
        mse = mean_squared_error(actual_aligned, forecast_aligned)
        mae = mean_absolute_error(actual_aligned, forecast_aligned)
        rmse = np.sqrt(mse)
        
        # Direction accuracy
        direction_correct = np.sum(
            np.sign(actual_aligned.diff()) == np.sign(forecast_aligned.diff())
        )
        direction_accuracy = direction_correct / len(actual_aligned)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy
        }


class RollingGARCH:
    """
    Rolling window GARCH model for time-varying volatility forecasting.
    """
    
    def __init__(self, window_size: int = 252, forecast_horizon: int = 1):
        """
        Initialize rolling GARCH model.
        
        Args:
            window_size: Size of rolling window
            forecast_horizon: Forecast horizon
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.forecasts = []
        self.actuals = []
        
    def fit_and_forecast(self, returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Fit rolling GARCH model and generate forecasts.
        
        Args:
            returns: Series of log returns
            
        Returns:
            Tuple of (forecasts, actuals)
        """
        forecasts = []
        actuals = []
        
        for i in range(self.window_size, len(returns)):
            # Training window
            train_returns = returns.iloc[i-self.window_size:i]
            
            # Test window
            test_returns = returns.iloc[i:i+self.forecast_horizon]
            
            try:
                # Fit GARCH model
                garch = GARCHModel()
                garch.fit(train_returns)
                
                # Generate forecast
                forecast = garch.forecast(self.forecast_horizon)
                forecasts.extend(forecast.values)
                
                # Calculate actual volatility
                actual_vol = test_returns.std() * np.sqrt(252)
                actuals.extend([actual_vol] * self.forecast_horizon)
                
            except Exception as e:
                logger.warning(f"Error in rolling window {i}: {str(e)}")
                forecasts.extend([np.nan] * self.forecast_horizon)
                actuals.extend([np.nan] * self.forecast_horizon)
        
        # Create series with proper dates
        forecast_dates = returns.index[self.window_size:]
        forecast_series = pd.Series(forecasts, index=forecast_dates)
        actual_series = pd.Series(actuals, index=forecast_dates)
        
        return forecast_series, actual_series


def compare_garch_models(returns: pd.Series, models: Dict[str, Tuple[int, int]]) -> Dict:
    """
    Compare different GARCH model specifications.
    
    Args:
        returns: Series of log returns
        models: Dictionary of model names to (p, q) tuples
        
    Returns:
        Dictionary with model comparison results
    """
    results = {}
    
    for name, (p, q) in models.items():
        try:
            garch = GARCHModel(p=p, q=q)
            garch.fit(returns)
            summary = garch.get_model_summary()
            
            results[name] = {
                'aic': summary.get('aic', np.nan),
                'bic': summary.get('bic', np.nan),
                'log_likelihood': summary.get('log_likelihood', np.nan),
                'params': summary.get('params', {})
            }
            
        except Exception as e:
            logger.error(f"Error fitting {name}: {str(e)}")
            results[name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Get sample data
    btc = yf.download('BTC-USD', start='2020-01-01', end='2023-01-01')
    returns = np.log(btc['Close'] / btc['Close'].shift(1))
    
    # Fit GARCH model
    garch = GARCHModel()
    garch.fit(returns.dropna())
    
    # Generate forecasts
    forecasts = garch.forecast(horizon=5)
    print(f"Volatility forecasts: {forecasts}") 