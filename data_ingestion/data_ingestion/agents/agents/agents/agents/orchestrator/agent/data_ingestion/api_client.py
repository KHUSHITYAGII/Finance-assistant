"""
API client for fetching market data from Alpha Vantage or Yahoo Finance
"""

import os
import yfinance as yf
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from loguru import logger

from config.settings import settings


class APIClient:
    """Client for fetching financial data from APIs"""
    
    def __init__(self):
        """Initialize the API client"""
        self.use_alpha_vantage = settings.USE_ALPHA_VANTAGE
        
        if self.use_alpha_vantage:
            self.alpha_vantage_key = settings.ALPHA_VANTAGE_API_KEY.get_secret_value()
            self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            self.fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
        
        self.asia_tech_tickers = settings.ASIA_TECH_TICKERS
    
    async def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get stock data for a specific ticker
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Dict containing stock data
        """
        try:
            if self.use_alpha_vantage:
                return await self._get_alpha_vantage_stock_data(ticker)
            else:
                return await self._get_yahoo_finance_stock_data(ticker)
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return {
                "ticker": ticker,
                "name": ticker,
                "price": 0.0,
                "change_percent": 0.0,
                "volume": 0,
                "error": str(e)
            }
    
    async def _get_alpha_vantage_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Get stock data from Alpha Vantage"""
        data, meta_data = self.ts.get_quote_endpoint(symbol=ticker)
        
        return {
            "ticker": ticker,
            "name": ticker,  # Alpha Vantage doesn't provide company name in this endpoint
            "price": float(data['05. price'][0]),
            "change_percent": float(data['10. change percent'][0].strip('%')),
            "volume": int(data['06. volume'][0])
        }
    
    async def _get_yahoo_finance_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Get stock data from Yahoo Finance"""
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="2d")
        
        if hist.empty:
            raise ValueError(f"No historical data available for {ticker}")
        
        # Calculate daily change
        if len(hist) >= 2:
            yesterday_close = hist['Close'].iloc[-2]
            today_close = hist['Close'].iloc[-1]
            change_percent = ((today_close - yesterday_close) / yesterday_close) * 100
        else:
            change_percent = 0.0
        
        return {
            "ticker": ticker,
            "name": info.get('shortName', ticker),
            "price": info.get('currentPrice', hist['Close'].iloc[-1]),
            "change_percent": change_percent,
            "volume": info.get('volume', hist['Volume'].iloc[-1])
        }
    
    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data for a specific ticker
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Dict containing earnings data
        """
        try:
            if self.use_alpha_vantage:
                return await self._get_alpha_vantage_earnings_data(ticker)
            else:
                return await self._get_yahoo_finance_earnings_data(ticker)
        except Exception as e:
            logger.error(f"Error fetching earnings data for {ticker}: {e}")
            return {
                "ticker": ticker,
                "name": ticker,
                "eps_estimate": 0.0,
                "eps_actual": 0.0,
                "surprise_percent": 0.0,
                "error": str(e)
            }
    
    async def _get_alpha_vantage_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data from Alpha Vantage"""
        data, meta_data = self.fd.get_company_earnings(symbol=ticker)
        
        # Get the most recent quarterly report
        quarterly_data = data['quarterlyEarnings'][0]
        
        return {
            "ticker": ticker,
            "name": ticker,  # Alpha Vantage doesn't provide company name in this endpoint
            "eps_estimate": float(quarterly_data['estimatedEPS']),
            "eps_actual": float(quarterly_data['reportedEPS']),
            "surprise_percent": float(quarterly_data['surprisePercentage'])
        }
    
    async def _get_yahoo_finance_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data from Yahoo Finance"""
        stock = yf.Ticker(ticker)
        
        # Get earnings information
        earnings = stock.earnings
        
        # Get the most recent quarterly earnings estimate
        calendar = stock.calendar
        
        if calendar is not None and 'Earnings Estimate' in calendar:
            eps_estimate = calendar['Earnings Estimate'].iloc[0]
        else:
            # If no estimates are available, use historical data
            if hasattr(earnings, 'iloc') and len(earnings) > 0:
                eps_estimate = earnings.iloc[-1]['Earnings']
            else:
                eps_estimate = 0.0
        
        # Get actual earnings (this is crude and would be better with real earnings surprise data)
        if hasattr(earnings, 'iloc') and len(earnings) > 0:
            eps_actual = earnings.iloc[-1]['Earnings']
        else:
            eps_actual = 0.0
        
        # Calculate surprise percentage
        if eps_estimate != 0:
            surprise_percent = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
        else:
            surprise_percent = 0.0
        
        return {
            "ticker": ticker,
            "name": stock.info.get('shortName', ticker),
            "eps_estimate": float(eps_estimate),
            "eps_actual": float(eps_actual),
            "surprise_percent": float(surprise_percent)
        }
    
    async def get_portfolio_allocation(self, region: str = "Asia", sector: str = "Technology") -> Dict[str, Any]:
        """
        Get portfolio allocation data for a specific region and sector
        This is a simulated function as we don't have real portfolio data
        
        Args:
            region: Geographic region
            sector: Market sector
            
        Returns:
            Dict containing portfolio allocation data
        """
        # In a real application, this would fetch data from a portfolio management system
        # Here we'll simulate the data
        
        # Simulate current allocation
        allocation_percent = 22.0  # As per the example in the assignment
        
        # Simulate previous allocation
        previous_percent = 18.0  # As per the example in the assignment
        
        return {
            "sector": sector,
            "region": region,
            "allocation_percent": allocation_percent,
            "previous_percent": previous_percent,
            "change": allocation_percent - previous_percent
        }
    
    async def get_asia_tech_stocks_data(self) -> List[Dict[str, Any]]:
        """
        Get data for all Asia tech stocks in the portfolio
        
        Returns:
            List of dicts containing stock data
        """
        results = []
        
        for ticker in self.asia_tech_tickers:
            stock_data = await self.get_stock_data(ticker)
            results.append(stock_data)
        
        return results
    
    async def get_asia_tech_earnings_data(self) -> List[Dict[str, Any]]:
        """
        Get earnings data for all Asia tech stocks in the portfolio
        
        Returns:
            List of dicts containing earnings data
        """
        results = []
        
        for ticker in self.asia_tech_tickers:
            earnings_data = await self.get_earnings_data(ticker)
            results.append(earnings_data)
        
        return results