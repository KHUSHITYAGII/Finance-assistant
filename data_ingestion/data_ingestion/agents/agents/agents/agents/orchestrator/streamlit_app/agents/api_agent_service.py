from fastapi import FastAPI, HTTPException, Query
from typing import Optional, Dict, List, Any, Union
import httpx
import pandas as pd
import yfinance as yf
from pydantic import BaseModel
import os
import json
import logging
from datetime import datetime, timedelta
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_agent")

# Initialize FastAPI app
app = FastAPI(title="API Agent Service", description="Financial data API agent service")

# Load API keys from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Base URLs for APIs
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Pydantic models for requests and responses
class StockDataRequest(BaseModel):
    symbols: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: Optional[str] = "1d"  # daily by default

class MarketNewsRequest(BaseModel):
    category: Optional[str] = "general"
    limit: Optional[int] = 10

class StockQuoteResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    timestamp: str
    
class EarningsResponse(BaseModel):
    symbol: str
    reported_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    surprise: Optional[float] = None
    surprise_percent: Optional[float] = None
    report_date: str
    
class PortfolioExposureRequest(BaseModel):
    region: str
    sector: str
    portfolio_data: Optional[Dict[str, float]] = None  # symbol: weight

class PortfolioExposureResponse(BaseModel):
    total_exposure: float
    previous_exposure: Optional[float] = None
    exposure_change: Optional[float] = None
    top_holdings: List[Dict[str, Any]]
    regional_breakdown: Dict[str, float]

# Helper Functions
async def fetch_alpha_vantage_data(function: str, symbol: str, **kwargs) -> Dict:
    """Fetch data from Alpha Vantage API"""
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
        **kwargs
    }
    
    logger.info(f"Fetching {function} data for {symbol} from Alpha Vantage")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(ALPHA_VANTAGE_BASE_URL, params=params)
        
        if response.status_code != 200:
            logger.error(f"Alpha Vantage API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Error fetching data from Alpha Vantage")
        
        data = response.json()
        
        # Handle API limit messages
        if "Note" in data:
            logger.warning(f"Alpha Vantage API limit note: {data['Note']}")
        
        # Handle errors
        if "Error Message" in data:
            logger.error(f"Alpha Vantage error: {data['Error Message']}")
            raise HTTPException(status_code=400, detail=data["Error Message"])
            
        return data

async def fetch_yahoo_finance_data(symbols: List[str], period: str = "1d", interval: str = "1d") -> Dict:
    """Fetch data using Yahoo Finance API"""
    logger.info(f"Fetching data for {symbols} from Yahoo Finance")
    
    try:
        if isinstance(symbols, list) and len(symbols) > 1:
            # For multiple symbols
            tickers = yf.Tickers(" ".join(symbols))
            data = {symbol: ticker.history(period=period, interval=interval) for symbol, ticker in tickers.tickers.items()}
        else:
            # For single symbol
            symbol = symbols[0] if isinstance(symbols, list) else symbols
            ticker = yf.Ticker(symbol)
            data = {symbol: ticker.history(period=period, interval=interval)}
        
        # Convert DataFrames to dictionaries
        result = {}
        for symbol, df in data.items():
            if not df.empty:
                df_dict = df.reset_index()
                df_dict['Date'] = df_dict['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                result[symbol] = df_dict.to_dict(orient='records')
            else:
                result[symbol] = []
                
        return result
    except Exception as e:
        logger.error(f"Yahoo Finance error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error fetching data from Yahoo Finance: {str(e)}")

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "api_agent", "timestamp": datetime.now().isoformat()}

@app.get("/quote/{symbol}")
async def get_stock_quote(symbol: str):
    """Get real-time stock quote"""
    try:
        # Try Yahoo Finance first
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1d")
        
        if hist.empty:
            raise ValueError(f"No data found for symbol {symbol}")
            
        last_price = hist['Close'].iloc[-1]
        prev_close = info.get('previousClose', hist['Close'].iloc[0])
        change = last_price - prev_close
        change_percent = (change / prev_close) * 100 if prev_close else 0
        
        return {
            "symbol": symbol,
            "price": last_price,
            "change": change,
            "change_percent": change_percent,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.warning(f"Failed to get quote from Yahoo Finance: {str(e)}, trying Alpha Vantage")
        
        # Fallback to Alpha Vantage
        try:
            data = await fetch_alpha_vantage_data("GLOBAL_QUOTE", symbol)
            quote = data.get("Global Quote", {})
            
            if not quote:
                raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
                
            return {
                "symbol": symbol,
                "price": float(quote.get("05. price", 0)),
                "change": float(quote.get("09. change", 0)),
                "change_percent": float(quote.get("10. change percent", "0%").strip('%')),
                "timestamp": quote.get("07. latest trading day", datetime.now().isoformat())
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting quote from Alpha Vantage: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting stock quote: {str(e)}")

@app.post("/historical-data")
async def get_historical_data(request: StockDataRequest):
    """Get historical stock data for multiple symbols"""
    try:
        # Parse dates
        end_date = datetime.now() if not request.end_date else datetime.strptime(request.end_date, "%Y-%m-%d")
        start_date = (end_date - timedelta(days=30)) if not request.start_date else datetime.strptime(request.start_date, "%Y-%m-%d")
        
        # Calculate period for Yahoo Finance
        days_diff = (end_date - start_date).days
        if days_diff <= 7:
            period = f"{days_diff}d"
        elif days_diff <= 30:
            period = f"{round(days_diff/7)}wk"
        elif days_diff <= 365:
            period = f"{round(days_diff/30)}mo"
        else:
            period = f"{round(days_diff/365)}y"
            
        # Get data from Yahoo Finance
        data = await fetch_yahoo_finance_data(request.symbols, period=period, interval=request.interval)
        return data
    except Exception as e:
        logger.error(f"Error getting historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting historical data: {str(e)}")

@app.get("/earnings/{symbol}")
async def get_earnings(symbol: str):
    """Get latest earnings data for a symbol"""
    try:
        # Try Yahoo Finance first
        ticker = yf.Ticker(symbol)
        earnings = ticker.earnings
        calendar = ticker.calendar
        
        latest_earnings = {}
        
        # Check if we have calendar data (upcoming earnings)
        if calendar is not None and not calendar.empty:
            try:
                latest_earnings = {
                    "symbol": symbol,
                    "reported_eps": None,  # Not reported yet
                    "estimated_eps": float(calendar["EPS Estimate"].iloc[0]) if "EPS Estimate" in calendar.columns else None,
                    "surprise": None,  # Not reported yet
                    "surprise_percent": None,  # Not reported yet
                    "report_date": calendar.index[0].strftime("%Y-%m-%d") if len(calendar.index) > 0 else None
                }
            except Exception as e:
                logger.warning(f"Error parsing calendar data: {str(e)}")
        
        # Check if we have historical earnings
        if earnings is not None and not earnings.empty and (not latest_earnings or latest_earnings["reported_eps"] is None):
            try:
                # Get the most recent earnings
                if not earnings.empty and "Earnings" in earnings.columns:
                    latest_idx = earnings.index[-1]
                    reported = float(earnings["Earnings"].iloc[-1])
                    estimated = float(earnings["Earnings Estimate"].iloc[-1]) if "Earnings Estimate" in earnings.columns else None
                    
                    surprise = reported - estimated if estimated is not None else None
                    surprise_percent = (surprise / estimated) * 100 if surprise is not None and estimated != 0 else None
                    
                    latest_earnings = {
                        "symbol": symbol,
                        "reported_eps": reported,
                        "estimated_eps": estimated,
                        "surprise": surprise,
                        "surprise_percent": surprise_percent,
                        "report_date": latest_idx.strftime("%Y-%m-%d")
                    }
            except Exception as e:
                logger.warning(f"Error parsing earnings data: {str(e)}")
        
        # If we still don't have earnings data, fallback to Alpha Vantage
        if not latest_earnings:
            logger.info(f"No earnings data from Yahoo Finance for {symbol}, trying Alpha Vantage")
            data = await fetch_alpha_vantage_data("EARNINGS", symbol)
            
            if "quarterlyEarnings" in data and data["quarterlyEarnings"]:
                latest = data["quarterlyEarnings"][0]
                reported = float(latest.get("reportedEPS", 0))
                estimated = float(latest.get("estimatedEPS", 0))
                surprise = float(latest.get("surprise", 0))
                surprise_percent = float(latest.get("surprisePercentage", 0))
                
                latest_earnings = {
                    "symbol": symbol,
                    "reported_eps": reported,
                    "estimated_eps": estimated,
                    "surprise": surprise,
                    "surprise_percent": surprise_percent,
                    "report_date": latest.get("reportedDate", datetime.now().strftime("%Y-%m-%d"))
                }
        
        if not latest_earnings:
            raise HTTPException(status_code=404, detail=f"No earnings data found for {symbol}")
            
        return latest_earnings
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting earnings data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting earnings data: {str(e)}")

@app.post("/portfolio/exposure")
async def get_portfolio_exposure(request: PortfolioExposureRequest):
    """Calculate portfolio exposure to specific region and sector"""
    try:
        # If portfolio data not provided, use a default example
        portfolio_data = request.portfolio_data or {
            "AAPL": 0.05, "MSFT": 0.05, "GOOGL": 0.04, "AMZN": 0.04,
            "TSM": 0.03, "9988.HK": 0.02, "005930.KS": 0.03, "6758.T": 0.02
        }
        
        result = {
            "total_exposure": 0.0,
            "previous_exposure": 0.0,  # This would be retrieved from database in real implementation
            "top_holdings": [],
            "regional_breakdown": {}
        }
        
        # Define regions and corresponding exchanges/suffix patterns
        regions = {
            "Asia": [".KS", ".T", ".HK", ".TW", ".SZ", ".SS", "0", "6", "9"],  # Korean, Japanese, Hong Kong, Taiwan, China A-shares
            "US": ["NASDAQ:", "NYSE:", "", ".US"],  # US exchanges
            "Europe": [".L", ".PA", ".F", ".MI", ".AMS"]  # London, Paris, Frankfurt, Milan, Amsterdam
        }
        
        # Get info for all symbols
        all_info = {}
        for symbol in portfolio_data.keys():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                all_info[symbol] = info
            except Exception as e:
                logger.warning(f"Could not get info for {symbol}: {str(e)}")
                all_info[symbol] = {}
        
        # Calculate exposure
        tech_sector_terms = ["technology", "tech", "semiconductor", "software", "hardware", "electronics"]
        
        regional_exposure = {region: 0.0 for region in regions}
        total_exposure = 0.0
        
        for symbol, weight in portfolio_data.items():
            info = all_info.get(symbol, {})
            
            # Check sector match
            sector = info.get("sector", "").lower()
            industry = info.get("industry", "").lower()
            
            is_tech = any(term in sector.lower() for term in tech_sector_terms) or \
                     any(term in industry.lower() for term in tech_sector_terms)
            
            # Check region match
            symbol_region = None
            for region_name, patterns in regions.items():
                if any(symbol.endswith(pattern) for pattern in patterns) or \
                   any(pattern in symbol for pattern in patterns):
                    symbol_region = region_name
                    break
            
            # If we couldn't determine region from symbol, use country from info
            if not symbol_region and "country" in info:
                country = info["country"]
                if country in ["China", "Japan", "South Korea", "Taiwan", "Hong Kong", "Singapore", "India", "Indonesia", "Malaysia", "Thailand", "Vietnam"]:
                    symbol_region = "Asia"
                elif country in ["United States"]:
                    symbol_region = "US"
                elif country in ["United Kingdom", "France", "Germany", "Italy", "Spain", "Netherlands", "Switzerland", "Sweden", "Norway", "Denmark", "Finland"]:
                    symbol_region = "Europe"
            
            # Calculate exposure if both sector and region match
            if is_tech and symbol_region == request.region:
                stock_exposure = weight
                total_exposure += stock_exposure
                regional_exposure[symbol_region] += stock_exposure
                
                # Add to top holdings
                result["top_holdings"].append({
                    "symbol": symbol,
                    "name": info.get("shortName", symbol),
                    "weight": weight,
                    "sector": sector,
                    "industry": industry,
                    "region": symbol_region
                })
        
        # Sort top holdings by weight
        result["top_holdings"] = sorted(result["top_holdings"], key=lambda x: x["weight"], reverse=True)
        
        # Set regional breakdown
        result["regional_breakdown"] = regional_exposure
        
        # Total exposure
        result["total_exposure"] = total_exposure
        
        # Mock previous exposure (in real implementation, this would come from database)
        result["previous_exposure"] = total_exposure * 0.9  # Assuming 10% increase from previous day
        result["exposure_change"] = total_exposure - result["previous_exposure"]
        
        return result
    except Exception as e:
        logger.error(f"Error calculating portfolio exposure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating portfolio exposure: {str(e)}")

@app.get("/market-news")
async def get_market_news(category: str = "general", limit: int = 10):
    """Get latest market news"""
    try:
        # We'll use Yahoo Finance for news (in a real implementation, you might use a news API)
        tickers = yf.Tickers(" ".join(["^GSPC", "^DJI", "^IXIC"]))  # S&P 500, Dow, Nasdaq
        
        # Collect news from all tickers
        all_news = []
        for ticker_symbol, ticker in tickers.tickers.items():
            try:
                news = ticker.news
                if news:
                    for item in news:
                        item["ticker"] = ticker_symbol
                        all_news.append(item)
            except Exception as e:
                logger.warning(f"Error getting news for {ticker_symbol}: {str(e)}")
        
        # Sort by publish time and limit results
        all_news = sorted(all_news, key=lambda x: x.get("providerPublishTime", 0), reverse=True)[:limit]
        
        # Format the news
        formatted_news = []
        for item in all_news:
            formatted_news.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("link", ""),
                "source": item.get("publisher", ""),
                "published_at": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat(),
                "related_tickers": item.get("relatedTickers", [])
            })
        
        return {"news": formatted_news}
    except Exception as e:
        logger.error(f"Error getting market news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting market news: {str(e)}")

@app.get("/market-sentiment")
async def get_market_sentiment(region: str = "global"):
    """Get market sentiment indicators"""
    try:
        # Define indices to check based on region
        indices = {
            "global": ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^N225"],
            "us": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
            "asia": ["^N225", "^HSI", "000001.SS", "^STI", "^AXJO"],
            "europe": ["^FTSE", "^GDAXI", "^FCHI", "^STOXX50E"]
        }
        
        region_indices = indices.get(region.lower(), indices["global"])
        
        # Get data for all indices
        data = await fetch_yahoo_finance_data(region_indices, period="5d")
        
        # Calculate sentiment indicators
        sentiment_score = 0
        total_indices = len(region_indices)
        region_data = {}
        
        for symbol, history in data.items():
            if not history:
                continue
                
            # Get most recent and previous day
            try:
                latest = history[-1]
                previous = history[-2] if len(history) > 1 else history[0]
                
                # Calculate daily change
                close = float(latest["Close"])
                prev_close = float(previous["Close"])
                change = close - prev_close
                change_percent = (change / prev_close) * 100 if prev_close else 0
                
                # Simple sentiment calculation
                if change_percent > 1.5:
                    sentiment = "very bullish"
                    score = 2
                elif change_percent > 0:
                    sentiment = "bullish"
                    score = 1
                elif change_percent > -1.5:
                    sentiment = "bearish"
                    score = -1
                else:
                    sentiment = "very bearish"
                    score = -2
                    
                sentiment_score += score
                
                # Store data for this index
                index_name = symbol
                if symbol == "^GSPC":
                    index_name = "S&P 500"
                elif symbol == "^DJI":
                    index_name = "Dow Jones"
                elif symbol == "^IXIC":
                    index_name = "NASDAQ"
                elif symbol == "^FTSE":
                    index_name = "FTSE 100"
                elif symbol == "^N225":
                    index_name = "Nikkei 225"
                elif symbol == "^HSI":
                    index_name = "Hang Seng"
                elif symbol == "000001.SS":
                    index_name = "Shanghai Composite"
                
                region_data[index_name] = {
                    "symbol": symbol,
                    "price": close,
                    "change": change,
                    "change_percent": change_percent,
                    "sentiment": sentiment
                }
            except Exception as e:
                logger.warning(f"Error calculating sentiment for {symbol}: {str(e)}")
        
        # Overall sentiment
        if total_indices > 0:
            avg_sentiment = sentiment_score / total_indices
            
            if avg_sentiment > 1:
                overall_sentiment = "very bullish"
            elif avg_sentiment > 0:
                overall_sentiment = "bullish"
            elif avg_sentiment > -1:
                overall_sentiment = "neutral with bearish tilt"
            elif avg_sentiment > -1.5:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "very bearish"
        else:
            overall_sentiment = "neutral"
        
        # Get additional sentiment indicators
        # In a real implementation, you'd also check things like VIX, put/call ratios, etc.
        try:
            vix_data = await fetch_yahoo_finance_data(["^VIX"], period="5d")
            if vix_data.get("^VIX") and vix_data["^VIX"]:
                latest_vix = float(vix_data["^VIX"][-1]["Close"])
                prev_vix = float(vix_data["^VIX"][-2]["Close"]) if len(vix_data["^VIX"]) > 1 else latest_vix
                
                vix_change = latest_vix - prev_vix
                vix_sentiment = "anxious" if latest_vix > 25 else ("cautious" if latest_vix > 15 else "complacent")
                
                # Refine overall sentiment with VIX information
                if latest_vix > 30 and overall_sentiment.startswith("very bullish"):
                    overall_sentiment = "bullish with high volatility concerns"
                elif latest_vix > 30 and "bearish" in overall_sentiment:
                    overall_sentiment = "very bearish with panic signals"
                elif vix_change > 2 and "bullish" in overall_sentiment:
                    overall_sentiment = "cautiously bullish"
        except Exception as e:
            logger.warning(f"Error getting VIX data: {str(e)}")
        
        # Return sentiment analysis
        return {
            "region": region,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score / total_indices if total_indices > 0 else 0,
            "market_indices": region_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting market sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting market sentiment: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)