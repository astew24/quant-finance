"""
Crypto Data Collector Module

This module handles data collection from Binance API for BTC and ETH price data.
Provides functionality for historical data fetching and real-time data streaming.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataCollector:
    """
    A class to collect cryptocurrency price data from Binance exchange.
    
    Attributes:
        exchange: Binance exchange instance
        symbols: List of trading pairs to collect data for
    """
    
    def __init__(self, symbols: Optional[List[str]] = None, exchange: Optional[Any] = None) -> None:
        """
        Initialize the data collector.
        
        Args:
            symbols: List of trading pairs (default: ['BTC/USDT', 'ETH/USDT'])
            exchange: Optional exchange client (used for testing or custom clients)
        """
        if exchange is None:
            try:
                import ccxt  # type: ignore
            except Exception as exc:
                raise ImportError("ccxt is required to create a default exchange client") from exc
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
        else:
            self.exchange = exchange
        
        self.symbols: List[str] = symbols or ['BTC/USDT', 'ETH/USDT']
        if not all(isinstance(symbol, str) and symbol for symbol in self.symbols):
            raise ValueError("All symbols must be non-empty strings")
        logger.info(f"Initialized CryptoDataCollector for symbols: {self.symbols}")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1d', 
                    since: Optional[datetime] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data ('1m', '5m', '1h', '1d', etc.)
            since: Start date for data collection
            limit: Maximum number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        if not timeframe:
            raise ValueError("timeframe must be a non-empty string")
        if limit <= 0:
            raise ValueError("limit must be positive")

        try:
            # Convert datetime to timestamp if provided
            since_timestamp: Optional[int] = None
            if since:
                since_timestamp = int(since.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv: List[List[float]] = self.exchange.fetch_ohlcv(
                symbol, timeframe, since_timestamp, limit
            )
            
            # Convert to DataFrame
            df: pd.DataFrame = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_historical_data(self, days_back: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all symbols.
        
        Args:
            days_back: Number of days to go back in history
            
        Returns:
            Dictionary mapping symbols to their historical data
        """
        if days_back <= 0:
            raise ValueError("days_back must be positive")
        since: datetime = datetime.now() - timedelta(days=days_back)
        data: Dict[str, pd.DataFrame] = {}
        
        for symbol in self.symbols:
            logger.info(f"Fetching historical data for {symbol}")
            df: pd.DataFrame = self.fetch_ohlcv(symbol, '1d', since)
            if not df.empty:
                data[symbol] = df
                time.sleep(0.1)  # Rate limiting
        
        return data
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate log returns from price data.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series of log returns
        """
        if 'close' not in df.columns:
            raise KeyError("Input DataFrame must contain a 'close' column")
        if (df['close'] <= 0).any():
            raise ValueError("Prices must be positive to compute log returns")
        return np.log(df['close'] / df['close'].shift(1))
    
    def calculate_volatility(self, returns: pd.Series, window: int = 30) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Series of returns
            window: Rolling window size
            
        Returns:
            Series of rolling volatility
        """
        if window <= 1:
            raise ValueError("window must be greater than 1")
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def get_market_data(self, days_back: int = 365) -> Dict[str, Dict[str, pd.Series]]:
        """
        Get comprehensive market data including prices, returns, and volatility.
        
        Args:
            days_back: Number of days to go back
            
        Returns:
            Dictionary with processed market data for each symbol
        """
        raw_data: Dict[str, pd.DataFrame] = self.fetch_historical_data(days_back)
        market_data: Dict[str, Dict[str, pd.Series]] = {}
        
        for symbol, df in raw_data.items():
            if df.empty:
                continue
                
            returns: pd.Series = self.calculate_returns(df)
            volatility: pd.Series = self.calculate_volatility(returns)
            
            market_data[symbol] = {
                'prices': df['close'],
                'returns': returns,
                'volatility': volatility,
                'volume': df['volume']
            }
            
            logger.info(f"Processed market data for {symbol}")
        
        return market_data


def save_data_to_csv(data: Dict[str, pd.DataFrame], output_dir: str = 'data') -> None:
    """
    Save collected data to CSV files.
    
    Args:
        data: Dictionary of DataFrames to save
        output_dir: Directory to save files in
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, df in data.items():
        if df.empty:
            continue
        filename: str = f"{symbol.replace('/', '_')}_data.csv"
        filepath: str = os.path.join(output_dir, filename)
        df.to_csv(filepath)
        logger.info(f"Saved {symbol} data to {filepath}")


if __name__ == "__main__":
    # Example usage
    collector = CryptoDataCollector()
    data = collector.get_market_data(days_back=365)
    save_data_to_csv(data) 
