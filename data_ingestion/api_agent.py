import os
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Finance API Agent")

# Dictionary mapping regions to major indices and tech stocks
REGION_MAPPINGS = {
    "asia": {
        "indices": ["^NIKKEI", "^HSI", "^SENSEX", "^KOSPI"],
        "tech_stocks": ["9984.T", "9988.HK", "6758.T", "000660.KS", "066570.KS", "2330.TW"]
    },
    "us": {
        "indices": ["^GSPC", "^DJI", "^IXIC"],
        "tech_stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    },
    "europe": {
        "indices": ["^FTSE", "^GDAXI", "^FCHI"],
        "tech_stocks": ["ASML.AS", "SAP.DE", "STM.PA", "NOK.HE", "ERIC-B.ST"]
    }
}

# Stock ticker to company name mapping for tech stocks
TICKER_TO_COMPANY = {
    # Asia
    "9984.T": "SoftBank Group",
    "9988.HK": "Alibaba",
    "6758.T": "Sony",
    "000660.KS": "SK Hynix",
    "066570.KS": "LG Electronics",
    "2330.TW": "TSMC",
    # US
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NVDA": "NVIDIA",
    # Europe
    "ASML.AS": "ASML Holding",
    "SAP.DE": "SAP",
    "STM.PA": "STMicroelectronics",
    "NOK.HE": "Nokia",
    "ERIC-B.ST": "Ericsson"
}

class MarketDataRequest(BaseModel):
    query: str
    region: Optional[str] = None
    tickers: Optional[List[str]] = None
    days: Optional[int] = 2

@app.post("/get_market_data")
async def get_market_data(request: MarketDataRequest):
    """Get market data based on the query."""
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
        
        # Get tech stocks for the region
        if request.tickers:
            tickers = request.tickers
        else:
            tickers = REGION_MAPPINGS.get(region, {}).get("tech_stocks", [])
        
        # Get data for the region's indices
        indices = REGION_MAPPINGS.get(region, {}).get("indices", [])
        
        # Fetch market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)
        
        # Get stock data
        stock_data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Calculate daily returns
                    if len(hist) > 1:
                        daily_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                    else:
                        daily_return = 0
                    
                    # Get company info
                    company_name = TICKER_TO_COMPANY.get(ticker, ticker)
                    
                    # Try to get earnings data
                    try:
                        earnings = stock.earnings
                        recent_earnings = None
                        if not earnings.empty and len(earnings) > 0:
                            recent_earnings = {
                                "date": str(earnings.index[-1]),
                                "revenue": float(earnings["Revenue"].iloc[-1]),
                                "earnings": float(earnings["Earnings"].iloc[-1])
                            }
                    except Exception as e:
                        logger.warning(f"Failed to get earnings for {ticker}: {e}")
                        recent_earnings = None
                    
                    stock_data[ticker] = {
                        "name": company_name,
                        "current_price": float(hist['Close'].iloc[-1]),
                        "daily_change_pct": float(daily_return),
                        "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist else None,
                        "recent_earnings": recent_earnings
                    }
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        
        # Get index data
        index_data = {}
        for index in indices:
            try:
                idx = yf.Ticker(index)
                hist = idx.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Calculate daily returns
                    if len(hist) > 1:
                        daily_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                    else:
                        daily_return = 0
                    
                    index_data[index] = {
                        "current_level": float(hist['Close'].iloc[-1]),
                        "daily_change_pct": float(daily_return)
                    }
            except Exception as e:
                logger.error(f"Error fetching data for index {index}: {e}")
        
        # Calculate portfolio exposure
        # This is mocked for the purpose of the assignment
        # In a real system, you would calculate this based on actual portfolio holdings
        portfolio_data = {
            "region_allocation": {
                "asia": 22.0,  # Percentage of AUM
                "us": 45.0,
                "europe": 33.0
            },
            "yesterday_allocation": {
                "asia": 18.0,  # Previous day's allocation
                "us": 47.0,
                "europe": 35.0
            }
        }
        
        # Determine region sentiment based on index performances
        sentiment = "neutral"
        avg_change = 0
        if index_data:
            changes = [data["daily_change_pct"] for data in index_data.values()]
            avg_change = sum(changes) / len(changes) if changes else 0
            
            if avg_change > 1.0:
                sentiment = "positive"
            elif avg_change < -1.0:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        # Identify earnings surprises
        earnings_surprises = []
        for ticker, data in stock_data.items():
            # This is simplified - in a real system, you would compare actual earnings to estimates
            if data["daily_change_pct"] > 3.0:
                earnings_surprises.append({
                    "ticker": ticker,
                    "name": data["name"],
                    "change": data["daily_change_pct"],
                    "surprise": "positive"
                })
            elif data["daily_change_pct"] < -3.0:
                earnings_surprises.append({
                    "ticker": ticker,
                    "name": data["name"],
                    "change": data["daily_change_pct"],
                    "surprise": "negative"
                })
        
        # Prepare response
        response = {
            "success": True,
            "data": {
                "timestamp": datetime.now().isoformat(),
                "region": region,
                "portfolio": portfolio_data,
                "stocks": stock_data,
                "indices": index_data,
                "regional_sentiment": {
                    "overall": sentiment,
                    "avg_index_change": avg_change
                },
                "earnings_surprises": earnings_surprises
            }
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error in get_market_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve market data: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)