import numpy as np
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
app = FastAPI(title="Finance Analysis Agent")

class AnalysisRequest(BaseModel):
    query: str
    market_data: Dict[str, Any]
    news_data: Dict[str, Any]
    retrieval_data: Optional[Dict[str, Any]] = None

def calculate_risk_exposure(market_data):
    """Calculate risk exposure metrics."""
    try:
        portfolio = market_data.get("portfolio", {})
        region_allocation = portfolio.get("region_allocation", {})
        yesterday_allocation = portfolio.get("yesterday_allocation", {})
        
        # Calculate risk metrics
        risk_metrics = {}
        
        # Asia tech allocation change
        asia_allocation = region_allocation.get("asia", 0)
        asia_prev_allocation = yesterday_allocation.get("asia", 0)
        asia_allocation_change = asia_allocation - asia_prev_allocation
        
        # Calculate standard deviation of returns as volatility measure
        volatility = {}
        stocks = market_data.get("stocks", {})
        for region, alloc in region_allocation.items():
            region_stocks = [s for ticker, s in stocks.items() if s.get("region", "") == region]
            if region_stocks:
                returns = [s.get("daily_change_pct", 0) for s in region_stocks]
                volatility[region] = np.std(returns) if returns else 0
            else:
                volatility[region] = 0
        
        # Calculate correlation between different regions
        # Simplified for this example
        correlation = {
            "asia_us": 0.65,
            "asia_europe": 0.58,
            "us_europe": 0.72
        }
        
        # Extract earnings surprises
        earnings_surprises = market_data.get("earnings_surprises", [])
        
        risk_metrics = {
            "region_allocation": region_allocation,
            "allocation_changes": {
                "asia": asia_allocation_change,
                "us": region_allocation.get("us", 0) - yesterday_allocation.get("us", 0),
                "europe": region_allocation.get("europe", 0) - yesterday_allocation.get("europe", 0)
            },
            "volatility": volatility,
            "correlation": correlation,
            "earnings_impacts": earnings_surprises
        }
        
        return risk_metrics
    except Exception as e:
        logger.error(f"Error calculating risk exposure: {e}")
        return {}

def analyze_sentiment(market_data, news_data):
    """Analyze market sentiment based on market data and news."""
    try:
        # Extract sentiment indicators
        regional_sentiment = market_data.get("regional_sentiment", {})
        indices = market_data.get("indices", {})
        news = news_data.get("general_news", [])
        
        # Count positive and negative news mentions
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for news_item in news:
            title = news_item.get("title", "").lower()
            if any(word in title for word in ["rise", "gain", "jump", "positive", "beat", "surge"]):
                positive_count += 1
            elif any(word in title for word in ["fall", "drop", "decline", "negative", "miss", "plunge"]):
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate average index changes
        avg_index_change = 0
        if indices:
            changes = [data.get("daily_change_pct", 0) for data in indices.values()]
            avg_index_change = sum(changes) / len(changes) if changes else 0
        
        # Combine sentiment indicators
        sentiment_score = 0
        if positive_count > negative_count:
            sentiment_score += 1
        elif negative_count > positive_count:
            sentiment_score -= 1
        
        if avg_index_change > 0:
            sentiment_score += 1
        elif avg_index_change < 0:
            sentiment_score -= 1
        
        # Determine overall sentiment
        if sentiment_score >= 2:
            overall_sentiment = "strongly positive"
        elif sentiment_score == 1:
            overall_sentiment = "positive"
        elif sentiment_score == 0:
            overall_sentiment = "neutral"
        elif sentiment_score == -1:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "strongly negative"
        
        # Add yield context for cautionary tilt if present
        yield_context = ""
        for news_item in news:
            if "yield" in news_item.get("title", "").lower() and any(word in news_item.get("title", "").lower() for word in ["rise", "increase", "higher"]):
                yield_context = "cautionary tilt due to rising yields"
                break
        
        sentiment_analysis = {
            "overall": overall_sentiment,
            "context": yield_context if yield_context else None,
            "news_sentiment": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count
            },
            "market_indicators": {
                "avg_index_change": avg_index_change
            }
        }
        
        return sentiment_analysis
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return {"overall": "neutral"}

def analyze_earnings_surprises(market_data, news_data):
    """Analyze earnings surprises."""
    try:
        earnings_data = news_data.get("earnings_data", {})
        earnings_surprises = []
        
        for ticker, data in earnings_data.items():
            surprise_pct = data.get("surprise_pct", 0)
            if abs(surprise_pct) >= 1.0:
                company_name = None
                # Try to find company name in market data
                stocks = market_data.get("stocks", {})
                if ticker in stocks:
                    company_name = stocks[ticker].get("name", ticker)
                
                earnings_surprises.append({
                    "ticker": ticker,
                    "company": company_name or ticker,
                    "surprise_pct": surprise_pct,
                    "direction": "beat" if surprise_pct > 0 else "missed"
                })
        
        return earnings_surprises
    except Exception as e:
        logger.error(f"Error analyzing earnings surprises: {e}")
        return []

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """Analyze financial data and generate insights."""
    try:
        # Extract relevant data
        market_data = request.market_data
        news_data = request.news_data
        
        # 1. Calculate risk exposure
        risk_exposure = calculate_risk_exposure(market_data)
        
        # 2. Analyze market sentiment
        sentiment_analysis = analyze_sentiment(market_data, news_data)
        
        # 3. Analyze earnings surprises
        earnings_surprises = analyze_earnings_surprises(market_data, news_data)
        
        # Prepare analysis response
        response = {
            "success": True,
            "data": {
                "timestamp": datetime.now().isoformat(),
                "risk_exposure": risk_exposure,
                "sentiment": sentiment_analysis,
                "earnings_surprises": earnings_surprises,
                "query": request.query
            }
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error in analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze data: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)