"""
Stock Data Collector Module

This module handles data collection for equity markets and factor construction.
Provides functionality for downloading stock data and building Fama-French factors.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """
    A class to collect stock market data and construct factors.
    
    Attributes:
        symbols: List of stock symbols to collect data for
        start_date: Start date for data collection
        end_date: End date for data collection
    """
    
    def __init__(self, symbols: List[str] = None, start_date: str = None, end_date: str = None):
        """
        Initialize the data collector.
        
        Args:
            symbols: List of stock symbols (default: S&P 500 components)
            start_date: Start date for data collection
            end_date: End date for data collection
        """
        self.symbols = symbols or self._get_sp500_symbols()
        self.start_date = start_date or '2020-01-01'
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Initialized StockDataCollector for {len(self.symbols)} symbols")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
    
    def _get_sp500_symbols(self) -> List[str]:
        """
        Get S&P 500 component symbols.
        
        Returns:
            List of S&P 500 symbols
        """
        # This is a simplified version - in practice you might want to use
        # a more comprehensive source for S&P 500 components
        sp500_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'JNJ', 'JPM', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
            'ADBE', 'NFLX', 'CRM', 'CMCSA', 'PFE', 'ABT', 'KO', 'TMO', 'AVGO',
            'COST', 'DHR', 'MRK', 'WMT', 'ACN', 'NEE', 'TXN', 'QCOM', 'HON',
            'LLY', 'UNP', 'LOW', 'ORCL', 'UPS', 'IBM', 'RTX', 'T', 'CAT',
            'SPGI', 'INTU', 'MS', 'GS', 'AMGN', 'SCHW', 'ISRG', 'VRTX',
            'ADI', 'GILD', 'REGN', 'BKNG', 'MDLZ', 'TJX', 'KLAC', 'MNST',
            'ADP', 'CME', 'PANW', 'CDNS', 'SNPS', 'MELI', 'ASML', 'CHTR',
            'MU', 'CTAS', 'MAR', 'ORLY', 'PAYX', 'ROST', 'IDXX', 'BIIB',
            'DXCM', 'ALGN', 'CPRT', 'FAST', 'VRSK', 'SGEN', 'WDAY', 'ODFL',
            'EXC', 'XEL', 'AEP', 'WEC', 'DTE', 'SO', 'DUK', 'D', 'NEE'
        ]
        return sp500_symbols[:50]  # Limit to top 50 for demonstration
    
    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch stock data for a given symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch data using yfinance
            stock = yf.Ticker(symbol)
            data = stock.history(start=self.start_date, end=self.end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all symbols.
        
        Returns:
            Dictionary mapping symbols to their data
        """
        data = {}
        
        for symbol in self.symbols:
            logger.info(f"Fetching data for {symbol}")
            df = self.fetch_stock_data(symbol)
            if not df.empty:
                data[symbol] = df
                time.sleep(0.1)  # Rate limiting
        
        return data
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate log returns from price data.
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            Series of log returns
        """
        return np.log(df['Close'] / df['Close'].shift(1))
    
    def get_returns_data(self) -> pd.DataFrame:
        """
        Get returns data for all symbols.
        
        Returns:
            DataFrame with returns for all symbols
        """
        market_data = self.fetch_market_data()
        returns_data = {}
        
        for symbol, df in market_data.items():
            if not df.empty:
                returns = self.calculate_returns(df)
                returns_data[symbol] = returns
        
        return pd.DataFrame(returns_data)


class FactorConstructor:
    """
    A class to construct Fama-French factors and momentum signals.
    """
    
    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize factor constructor.
        
        Args:
            start_date: Start date for factor construction
            end_date: End date for factor construction
        """
        self.start_date = start_date or '2020-01-01'
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
    def construct_market_factor(self, returns_data: pd.DataFrame) -> pd.Series:
        """
        Construct market factor (equally weighted market return).
        
        Args:
            returns_data: DataFrame with stock returns
            
        Returns:
            Market factor series
        """
        # Calculate equally weighted market return
        market_factor = returns_data.mean(axis=1)
        return market_factor
    
    def construct_size_factor(self, returns_data: pd.DataFrame, 
                            market_caps: Dict[str, float]) -> pd.Series:
        """
        Construct SMB (Small Minus Big) factor.
        
        Args:
            returns_data: DataFrame with stock returns
            market_caps: Dictionary mapping symbols to market caps
            
        Returns:
            SMB factor series
        """
        # This is a simplified SMB construction
        # In practice, you would use more sophisticated size classifications
        
        # Split stocks into small and big based on median market cap
        median_mcap = np.median(list(market_caps.values()))
        
        small_stocks = [s for s, mcap in market_caps.items() if mcap < median_mcap]
        big_stocks = [s for s, mcap in market_caps.items() if mcap >= median_mcap]
        
        # Calculate factor returns
        small_returns = returns_data[small_stocks].mean(axis=1)
        big_returns = returns_data[big_stocks].mean(axis=1)
        
        smb_factor = small_returns - big_returns
        return smb_factor
    
    def construct_value_factor(self, returns_data: pd.DataFrame,
                             book_to_market: Dict[str, float]) -> pd.Series:
        """
        Construct HML (High Minus Low) factor.
        
        Args:
            returns_data: DataFrame with stock returns
            book_to_market: Dictionary mapping symbols to B/M ratios
            
        Returns:
            HML factor series
        """
        # This is a simplified HML construction
        # In practice, you would use more sophisticated value classifications
        
        # Split stocks into high and low B/M based on median
        median_bm = np.median(list(book_to_market.values()))
        
        high_bm_stocks = [s for s, bm in book_to_market.items() if bm > median_bm]
        low_bm_stocks = [s for s, bm in book_to_market.items() if bm <= median_bm]
        
        # Calculate factor returns
        high_bm_returns = returns_data[high_bm_stocks].mean(axis=1)
        low_bm_returns = returns_data[low_bm_stocks].mean(axis=1)
        
        hml_factor = high_bm_returns - low_bm_returns
        return hml_factor
    
    def construct_momentum_factor(self, returns_data: pd.DataFrame, 
                                lookback_period: int = 252) -> pd.Series:
        """
        Construct momentum factor.
        
        Args:
            returns_data: DataFrame with stock returns
            lookback_period: Period for momentum calculation
            
        Returns:
            Momentum factor series
        """
        # Calculate cumulative returns over lookback period
        cumulative_returns = (1 + returns_data).rolling(lookback_period).apply(
            lambda x: np.prod(x) - 1
        )
        
        # Split stocks into high and low momentum based on median
        median_momentum = cumulative_returns.median(axis=1)
        
        momentum_factor = pd.Series(index=returns_data.index)
        
        for date in returns_data.index:
            if pd.isna(median_momentum[date]):
                momentum_factor[date] = np.nan
                continue
                
            high_momentum = cumulative_returns.loc[date] > median_momentum[date]
            low_momentum = cumulative_returns.loc[date] <= median_momentum[date]
            
            high_momentum_returns = returns_data.loc[date, high_momentum].mean()
            low_momentum_returns = returns_data.loc[date, low_momentum].mean()
            
            momentum_factor[date] = high_momentum_returns - low_momentum_returns
        
        return momentum_factor
    
    def construct_all_factors(self, returns_data: pd.DataFrame,
                             market_caps: Dict[str, float] = None,
                             book_to_market: Dict[str, float] = None) -> pd.DataFrame:
        """
        Construct all factors.
        
        Args:
            returns_data: DataFrame with stock returns
            market_caps: Dictionary mapping symbols to market caps
            book_to_market: Dictionary mapping symbols to B/M ratios
            
        Returns:
            DataFrame with all factors
        """
        factors = {}
        
        # Market factor
        factors['Market'] = self.construct_market_factor(returns_data)
        
        # Size factor (if market caps provided)
        if market_caps:
            factors['SMB'] = self.construct_size_factor(returns_data, market_caps)
        
        # Value factor (if B/M ratios provided)
        if book_to_market:
            factors['HML'] = self.construct_value_factor(returns_data, book_to_market)
        
        # Momentum factor
        factors['Momentum'] = self.construct_momentum_factor(returns_data)
        
        return pd.DataFrame(factors)


def save_data_to_csv(data: Dict[str, pd.DataFrame], output_dir: str = 'data'):
    """
    Save collected data to CSV files.
    
    Args:
        data: Dictionary of DataFrames to save
        output_dir: Directory to save files in
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, df in data.items():
        filename = f"{symbol}_data.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath)
        logger.info(f"Saved {symbol} data to {filepath}")


if __name__ == "__main__":
    # Example usage
    collector = StockDataCollector()
    returns_data = collector.get_returns_data()
    
    # Construct factors
    factor_constructor = FactorConstructor()
    factors = factor_constructor.construct_all_factors(returns_data)
    
    print("Data collection and factor construction complete!")
    print(f"Collected data for {len(returns_data.columns)} stocks")
    print(f"Constructed {len(factors.columns)} factors") 