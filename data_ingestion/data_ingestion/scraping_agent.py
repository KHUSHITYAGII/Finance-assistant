import os
import re
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import httpx
from bs4 import BeautifulSoup
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Finance Scraping Agent")

# News sources
NEWS_SOURCES = {
    "asia": [
        "https://finance.yahoo.com/topic/asia-economy",
        "https://www.cnbc.com/asia-economy/"
    ],
    "us": [
        "https://finance.yahoo.com/topic/economic-news",
        "https://www.cnbc.com/economy/"
    ],
    "europe": [
        "https://finance.yahoo.com/topic/europe-economy",
        "https://www.cnbc.com/europe-economy/"
    ]
}

class NewsRequest(BaseModel):
    query: str
    region: Optional[str] = None
    days: Optional[int] = 1

async def scrape_yahoo_finance(url: str) -> List[Dict[str, Any]]:
    """Scrape news from Yahoo Finance."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            
            # Find news items (this is a simplified approach)
            articles = soup.find_all('div', {'class': 'Ov(h)'})
            
            for article in articles[:10]:  # Limit to 10 articles for performance
                try:
                    title_elem = article.find('h3')
                    link_elem = article.find('a')
                    
                    if title_elem and link_elem:
                        title = title_elem.text
                        link = link_elem.get('href')
                        if not link.startswith('http'):
                            link = f"https://finance.yahoo.com{link}"
                        
                        # Extract date if available
                        date_elem = article.find('div', {'class': 'C(#959595)'})
                        date_str = date_elem.text if date_elem else "Unknown"
                        
                        news_items.append({
                            "title": title,
                            "link": link,
                            "source": "Yahoo Finance",
                            "date": date_str
                        })
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
            
            return news_items
    except Exception as e:
        logger.error(f"Error scraping Yahoo Finance: {e}")
        return []

async def scrape_cnbc(url: str) -> List[Dict[str, Any]]:
    """Scrape news from CNBC."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            
            # Find news items (this is a simplified approach)
            articles = soup.find_all('div', {'class': 'Card-titleContainer'})
            
            for article in articles[:10]:  # Limit to 10 articles for performance
                try:
                    title_elem = article.find('a')
                    
                    if title_elem:
                        title = title_elem.text
                        link = title_elem.get('href')
                        if not link.startswith('http'):
                            link = f"https://www.cnbc.com{link}"
                        
                        # Extract date if available (simplified)
                        date_str = "Today"  # Simplified date handling
                        
                        news_items.append({
                            "title": title,
                            "link": link,
                            "source": "CNBC",
                            "date": date_str
                        })
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
            
            return news_items
    except Exception as e:
        logger.error(f"Error scraping CNBC: {e}")
        return []

async def scrape_earnings_news(ticker: str) -> List[Dict[str, Any]]:
    """Scrape earnings news for a specific ticker."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            
            # Find news items related to earnings
            articles = soup.find_all('div', {'class': 'Ov(h)'})
            
            for article in articles[:5]:  # Limit to 5 earnings-related articles
                try:
                    title_elem = article.find('h3')
                    
                    if title_elem and ('earnings' in title_elem.text.lower() or 'report' in title_elem.text.lower()):
                        link_elem = article.find('a')
                        link = link_elem.get('href') if link_elem else ""
                        if not link.startswith('http'):
                            link = f"https://finance.yahoo.com{link}"
                        
                        news_items.append({
                            "title": title_elem.text,
                            "link": link,
                            "source": "Yahoo Finance",
                            "ticker": ticker
                        })
                except Exception as e:
                    logger.warning(f"Error parsing earnings article: {e}")
            
            return news_items
    except Exception as e:
        logger.error(f"Error scraping earnings news for {ticker}: {e}")
        return []

@app.post("/get_news")
async def get_news(request: NewsRequest):
    """Get financial news based on the query."""
    try:
        # Extract region from query if not provided
        region = request.region
        if not region:
            query_lower = request.query.lower()
            if "asia" in query_lower:
                region = "asia"
            elif "us" in query_lower or "united states" in query_lower:
                region = "us"
            elif "europe" in query_lower:
                region = "europe"
            else:
                # Default to Asia based on the use case
                region = "asia"
        
        # Get news sources for the region
        sources = NEWS_SOURCES.get(region, [])
        
        # Extract tickers from query
        tickers = []
        if "tsmc" in request.query.lower():
            tickers.append("2330.TW")
        if "samsung" in request.query.lower():
            tickers.append("005930.KS")
        
        # Scrape news from all sources
        all_news = []
        
        # Scrape Yahoo Finance
        for source in sources:
            if "yahoo" in source:
                news = await scrape_yahoo_finance(source)
                all_news.extend(news)
            elif "cnbc" in source:
                news = await scrape_cnbc(source)
                all_news.extend(news)
        
        # Scrape earnings news for specified tickers
        earnings_news = []
        for ticker in tickers:
            news = await scrape_earnings_news(ticker)
            earnings_news.extend(news)
        
        # Filter and process news
        # For the demo, we'll mock some earnings data
        earnings_data = {
            "2330.TW": {  # TSMC
                "estimate": 100,
                "actual": 104,
                "surprise_pct": 4.0,
                "date": datetime.now().strftime("%Y-%m-%d")
            },
            "005930.KS": {  # Samsung
                "estimate": 150,
                "actual": 147,
                "surprise_pct": -2.0,
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        }
        
        # Prepare response
        response = {
            "success": True,
            "data": {
                "region": region,
                "general_news": all_news,
                "earnings_news": earnings_news,
                "earnings_data": earnings_data
            }
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error in get_news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve news: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)