"""
LSTM Model Module

This module implements LSTM-based volatility forecasting using TensorFlow/Keras.
Provides functionality for data preprocessing, model training, and forecasting.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LSTMModel:
    """
    LSTM model for volatility forecasting.
    
    Attributes:
        model: Trained LSTM model
        scaler: Data scaler
        sequence_length: Length of input sequences
        features: Number of features
    """
    
    def __init__(self, sequence_length: int = 30, features: int = 1, 
                 units: int = 50, dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            features: Number of features
            units: Number of LSTM units
            dropout: Dropout rate
        """
        self.sequence_length = sequence_length
        self.features = features
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input data array
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
            
        return np.array(X), np.array(y)
    
    def build_model(self) -> Sequential:
        """
        Build LSTM model architecture.
        
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            LSTM(units=self.units, return_sequences=True, 
                 input_shape=(self.sequence_length, self.features)),
            Dropout(self.dropout),
            LSTM(units=self.units, return_sequences=False),
            Dropout(self.dropout),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare data for LSTM training.
        
        Args:
            data: Input time series
            
        Returns:
            Tuple of (X_train, y_train, scaler)
        """
        # Remove NaN values
        data_clean = data.dropna()
        
        # Reshape for scaling
        data_reshaped = data_clean.values.reshape(-1, 1)
        
        # Scale the data
        data_scaled = self.scaler.fit_transform(data_reshaped)
        
        # Create sequences
        X, y = self.create_sequences(data_scaled)
        
        return X, y, self.scaler
    
    def fit(self, data: pd.Series, validation_split: float = 0.2, 
            epochs: int = 100, batch_size: int = 32, verbose: int = 1) -> 'LSTMModel':
        """
        Fit LSTM model to data.
        
        Args:
            data: Input time series
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Self for method chaining
        """
        try:
            # Prepare data
            X, y, scaler = self.prepare_data(data)
            
            # Build model
            self.model = self.build_model()
            
            # Train model
            self.history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                shuffle=False
            )
            
            logger.info("LSTM model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            raise
            
        return self
    
    def predict(self, data: pd.Series) -> pd.Series:
        """
        Generate predictions using trained model.
        
        Args:
            data: Input time series
            
        Returns:
            Series of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        try:
            # Prepare data
            data_clean = data.dropna()
            data_scaled = self.scaler.transform(data_clean.values.reshape(-1, 1))
            
            # Create sequences for prediction
            X_pred = []
            for i in range(self.sequence_length, len(data_scaled)):
                X_pred.append(data_scaled[i-self.sequence_length:i])
            
            X_pred = np.array(X_pred)
            
            # Generate predictions
            predictions_scaled = self.model.predict(X_pred)
            
            # Inverse transform
            predictions = self.scaler.inverse_transform(predictions_scaled)
            
            # Create series with proper dates
            pred_dates = data_clean.index[self.sequence_length:]
            return pd.Series(predictions.flatten(), index=pred_dates)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def forecast(self, data: pd.Series, horizon: int = 1) -> pd.Series:
        """
        Generate multi-step forecasts.
        
        Args:
            data: Input time series
            horizon: Forecast horizon
            
        Returns:
            Series of forecasts
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
            
        try:
            # Get the last sequence
            data_clean = data.dropna()
            last_sequence = data_clean.values[-self.sequence_length:].reshape(-1, 1)
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            forecasts = []
            current_sequence = last_sequence_scaled.copy()
            
            for _ in range(horizon):
                # Reshape for prediction
                X_pred = current_sequence.reshape(1, self.sequence_length, self.features)
                
                # Predict next value
                pred_scaled = self.model.predict(X_pred, verbose=0)
                pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
                forecasts.append(pred)
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred_scaled
            
            # Create forecast dates
            last_date = data_clean.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            return pd.Series(forecasts, index=forecast_dates)
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {str(e)}")
            raise
    
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Align series
        common_index = actual.index.intersection(predicted.index)
        actual_aligned = actual.loc[common_index]
        predicted_aligned = predicted.loc[common_index]
        
        # Calculate metrics
        mse = mean_squared_error(actual_aligned, predicted_aligned)
        mae = mean_absolute_error(actual_aligned, predicted_aligned)
        rmse = np.sqrt(mse)
        
        # Direction accuracy
        direction_correct = np.sum(
            np.sign(actual_aligned.diff()) == np.sign(predicted_aligned.diff())
        )
        direction_accuracy = direction_correct / len(actual_aligned)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy
        }


class RollingLSTM:
    """
    Rolling window LSTM model for time-varying volatility forecasting.
    """
    
    def __init__(self, sequence_length: int = 30, window_size: int = 252, 
                 forecast_horizon: int = 1):
        """
        Initialize rolling LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            window_size: Size of rolling window
            forecast_horizon: Forecast horizon
        """
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.forecasts = []
        self.actuals = []
        
    def fit_and_forecast(self, data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Fit rolling LSTM model and generate forecasts.
        
        Args:
            data: Input time series
            
        Returns:
            Tuple of (forecasts, actuals)
        """
        forecasts = []
        actuals = []
        
        for i in range(self.window_size, len(data)):
            # Training window
            train_data = data.iloc[i-self.window_size:i]
            
            # Test window
            test_data = data.iloc[i:i+self.forecast_horizon]
            
            try:
                # Fit LSTM model
                lstm = LSTMModel(sequence_length=self.sequence_length)
                lstm.fit(train_data, epochs=50, verbose=0)
                
                # Generate forecast
                forecast = lstm.forecast(train_data, self.forecast_horizon)
                forecasts.extend(forecast.values)
                
                # Get actual values
                actuals.extend(test_data.values)
                
            except Exception as e:
                logger.warning(f"Error in rolling window {i}: {str(e)}")
                forecasts.extend([np.nan] * self.forecast_horizon)
                actuals.extend([np.nan] * self.forecast_horizon)
        
        # Create series with proper dates
        forecast_dates = data.index[self.window_size:]
        forecast_series = pd.Series(forecasts, index=forecast_dates)
        actual_series = pd.Series(actuals, index=forecast_dates)
        
        return forecast_series, actual_series


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Get sample data
    btc = yf.download('BTC-USD', start='2020-01-01', end='2023-01-01')
    returns = np.log(btc['Close'] / btc['Close'].shift(1))
    volatility = returns.rolling(window=30).std() * np.sqrt(252)
    
    # Fit LSTM model
    lstm = LSTMModel(sequence_length=30)
    lstm.fit(volatility.dropna())
    
    # Generate forecasts
    forecasts = lstm.forecast(volatility.dropna(), horizon=5)
    print(f"Volatility forecasts: {forecasts}") 