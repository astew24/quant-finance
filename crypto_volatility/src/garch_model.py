"""
GARCH(1,1) volatility modeling and forecasting.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GARCHModel:
    """Fits a GARCH(p,q) model and generates volatility forecasts."""

    def __init__(self, p: int = 1, q: int = 1, mean: str = 'Zero', vol: str = 'GARCH'):
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.model = None
        self.fitted_model = None
        self.params = None
        self.forecasts = None

    def fit(self, returns: pd.Series) -> 'GARCHModel':
        """Fit the model to a return series. Returns self for chaining."""
        returns_clean = returns.dropna()
        if len(returns_clean) < max(self.p, self.q) + 1:
            raise ValueError("Not enough data to fit GARCH model")

        self.model = arch_model(
            returns_clean, vol=self.vol, p=self.p, q=self.q,
            mean=self.mean, dist='normal'
        )
        self.fitted_model = self.model.fit(disp='off')
        self.params = self.fitted_model.params

        logger.info(f"GARCH({self.p},{self.q}) fitted -- params: {dict(self.params)}")
        return self

    def forecast(self, horizon: int = 1) -> pd.Series:
        """Generate h-step ahead volatility forecasts."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        fc = self.fitted_model.forecast(horizon=horizon)
        self.forecasts = np.sqrt(fc.variance.values[-1, :])

        last_date = self.fitted_model.resid.index[-1]
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                              periods=horizon, freq='D')
        return pd.Series(self.forecasts, index=dates)

    def get_model_summary(self) -> Dict:
        if self.fitted_model is None:
            return {}
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.loglikelihood,
            'params': self.params.to_dict(),
            'residuals': self.fitted_model.resid,
            'conditional_volatility': self.fitted_model.conditional_volatility,
        }

    def evaluate_forecasts(self, actual_vol: pd.Series,
                           forecast_vol: pd.Series) -> Dict[str, float]:
        """Compare forecast vs actual on overlapping dates."""
        common = actual_vol.index.intersection(forecast_vol.index)
        if len(common) == 0:
            raise ValueError("No overlapping dates between actual and forecast")

        a = actual_vol.loc[common]
        f = forecast_vol.loc[common]

        mse = mean_squared_error(a, f)
        mae = mean_absolute_error(a, f)
        rmse = np.sqrt(mse)

        if len(a) < 2:
            dir_acc = np.nan
        else:
            dir_acc = np.mean(np.sign(a.diff()) == np.sign(f.diff()))

        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'direction_accuracy': dir_acc}


class RollingGARCH:
    """Rolling-window GARCH for out-of-sample evaluation."""

    def __init__(self, window_size: int = 252, forecast_horizon: int = 1):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

    def fit_and_forecast(self, returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
        if len(returns) < self.window_size + self.forecast_horizon:
            raise ValueError("Not enough data for rolling window")

        forecasts, actuals, dates = [], [], []
        for i in range(self.window_size, len(returns) - self.forecast_horizon + 1):
            train = returns.iloc[i - self.window_size:i]
            test = returns.iloc[i:i + self.forecast_horizon]

            try:
                g = GARCHModel()
                g.fit(train)
                fc = g.forecast(self.forecast_horizon)
                forecasts.extend(fc.values)
                actual_vol = test.std() * np.sqrt(252)
                actuals.extend([actual_vol] * self.forecast_horizon)
                dates.extend(test.index)
            except Exception as e:
                logger.warning(f"Rolling window {i} failed: {e}")
                forecasts.extend([np.nan] * self.forecast_horizon)
                actuals.extend([np.nan] * self.forecast_horizon)

        return pd.Series(forecasts, index=dates), pd.Series(actuals, index=dates)


def compare_garch_models(returns: pd.Series, models: Dict[str, Tuple[int, int]]) -> Dict:
    """Fit multiple GARCH specs and compare AIC/BIC."""
    results = {}
    for name, (p, q) in models.items():
        try:
            g = GARCHModel(p=p, q=q)
            g.fit(returns)
            s = g.get_model_summary()
            results[name] = {
                'aic': s.get('aic', np.nan),
                'bic': s.get('bic', np.nan),
                'log_likelihood': s.get('log_likelihood', np.nan),
                'params': s.get('params', {}),
            }
        except Exception as e:
            logger.error(f"Failed to fit {name}: {e}")
            results[name] = {'error': str(e)}
    return results
