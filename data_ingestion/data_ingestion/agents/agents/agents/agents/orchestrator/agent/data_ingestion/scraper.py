"""
Web scraper for financial news and filings
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from typing import List, Dict, Any, Optional, Union
import asyncio
from datetime import datetime, timedelta
from loguru import logger

from config.settings import settings


class Scraper:
    """Scraper for financial news and filings"""
    
    def __init__(self):
        """Initialize the scraper"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # List of news sources to scrape
        self.news_sources = {
            "yahoo_finance": {
                "url": "https://finance.yahoo.com/topic/asia-tech/",
                "article_selector": "li.js-stream-content",
                "title_selector": "h3",
                "link_selector": "a",
                "summary_selector": "p"
            },
            "cnbc_asia": {
                "url": "https://www.cnbc.com/asia-tech/",
                "article_selector": ".Card-standardBreakerCard",
                "title_selector": ".Card-title",
                "link_selector": "a",
                "summary_selector": ".Card-description"
            }
        }
        
        # Filings sources
        self.filings_sources = {
            "sec_edgar": {
                "url": "https://www.sec.gov/edgar/searchedgar/companysearch.html"
            }
        }
    
    async def get_news(self, 
                       region: str = "Asia", 
                       sector: str = "Technology", 
                       days: int = 1) -> List[Dict[str, Any]]:
        """
        Scrape recent news related to a region and sector
        
        Args:
            region: Geographic region
            sector: Market sector
            days: Number of days to look back
            
        Returns:
            List of dicts containing news articles
        """
        all_news = []
        
        # Scrape each news source
        for source_name, source_config in self.news_sources.items():
            try:
                news = await self._scrape_news_source(
                    source_name,
                    source_config,
                    region,
                    sector,
                    days
                )
                all_news.extend(news)
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {e}")
        
        # Sort by date, newest first
        all_news.sort(key=lambda x: x.get('date', datetime.now()), reverse=True)
        
        return all_news
    
    async def _scrape_news_source(self,
                                 source_name: str,
                                 source_config: Dict[str, str],
                                 region: str,
                                 sector: str,
                                 days: int) -> List[Dict[str, Any]]:
        """Scrape a specific news source"""
        try:
            # In a real implementation, this would actually fetch and parse the website
            # For demo purposes, we'll return mock data
            return self._get_mock_news(source_name, region, sector, days)
        except Exception as e:
            logger.error(f"Error in _scrape_news_source for {source_name}: {e}")
            return []
    
    def _get_mock_news(self, source_name: str, region: str, sector: str, days: int) -> List[Dict[str, Any]]:
        """Generate mock news data for demonstration"""
        current_date = datetime.now()
        
        # Mock articles
        mock_articles = [
            {
                "title": "TSMC Reports Strong Quarterly Earnings, Beats Estimates by 4%",
                "summary": "Taiwan Semiconductor Manufacturing Company reported earnings above analyst expectations, citing strong demand for advanced chips used in AI applications.",
                "date": current_date - timedelta(hours=4),
                "source": "Yahoo Finance",
                "url": "https://finance.yahoo.com/news/tsmc-reports-strong-earnings",
                "sentiment": 0.8  # Positive sentiment
            },
            {
                "title": "Samsung Electronics Misses Q1 Expectations by 2% Amid Memory Price Pressure",
                "summary": "Samsung reported quarterly results below expectations as memory chip prices remained under pressure despite early signs of market recovery.",
                "date": current_date - timedelta(hours=8),
                "source": "CNBC Asia",
                "url": "https://www.cnbc.com/asia-tech/samsung-q1-results",
                "sentiment": -0.2  # Slightly negative sentiment
            },
            {
                "title": "Rising Bond Yields Creating Headwinds for Asian Tech Stocks",
                "summary": "Technology shares across Asia are facing pressure as rising global bond yields make growth stocks less attractive to investors.",
                "date": current_date - timedelta(hours=12),
                "source": "Yahoo Finance",
                "url": "https://finance.yahoo.com/news/bond-yields-asian-tech",
                "sentiment": -0.4  # Negative sentiment
            }
        ]
        
        # Filter by recency
        cutoff_date = current_date - timedelta(days=days)
        recent_articles = [article for article in mock_articles if article["date"] >= cutoff_date]
        
        return recent_articles
    
    async def get_filings(self, 
                         ticker: str, 
                         filing_type: str = "10-Q", 
                         limit: int = 1) -> List[Dict[str, Any]]:
        """
        Get recent SEC filings for a company
        
        Args:
            ticker: Company ticker symbol
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
            limit: Maximum number of filings to return
            
        Returns:
            List of dicts containing filing data
        """
        try:
            # In a real implementation, this would actually fetch and parse SEC EDGAR
            # For demo purposes, we'll return mock data
            return self._get_mock_filings(ticker, filing_type, limit)
        except Exception as e:
            logger.error(f"Error fetching filings for {ticker}: {e}")
            return []
    
    def _get_mock_filings(self, ticker: str, filing_type: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock filing data for demonstration"""
        current_date = datetime.now()
        
        # Mock filings
        mock_filings = [
            {
                "company": "Taiwan Semiconductor Manufacturing Company",
                "ticker": "TSM",
                "filing_type": "10-Q",
                "filing_date": current_date - timedelta(days=5),
                "period_end_date": current_date - timedelta(days=35),
                "url": "https://www.sec.gov/edgar/mockfilings/tsm-10q.html",
                "highlights": {
                    "revenue": "$15.2B, up 18% YoY",
                    "net_income": "$5.8B, up 22% YoY",
                    "eps": "$0.23, beating estimates of $0.22"
                }
            },
            {
                "company": "Samsung Electronics",
                "ticker": "005930.KS",
                "filing_type": "10-Q",
                "filing_date": current_date - timedelta(days=8),
                "period_end_date": current_date - timedelta(days=38),
                "url": "https://www.sec.gov/edgar/mockfilings/samsung-10q.html",
                "highlights": {
                    "revenue": "$58.9B, down 3% YoY",
                    "net_income": "$6.2B, down 8% YoY",
                    "eps": "$0.92, missing estimates of $0.94"
                }
            }
        ]
        
        # Filter by ticker and filing type
        filtered_filings = [
            filing for filing in mock_filings 
            if filing["ticker"] == ticker and filing["filing_type"] == filing_type
        ]
        
        # Limit results
        return filtered_filings[:limit]