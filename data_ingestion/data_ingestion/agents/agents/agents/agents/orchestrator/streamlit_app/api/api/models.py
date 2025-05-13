"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List


class MarketBriefRequest(BaseModel):
    """Request model for market brief"""
    question: str = Field(..., description="The question to generate a market brief for")


class MarketBriefResponse(BaseModel):
    """Response model for market brief"""
    brief: str = Field(..., description="The generated market brief")
    confidence: float = Field(..., description="Confidence score of the response", ge=0.0, le=1.0)
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources used to generate the brief"
    )
    audio_url: Optional[str] = Field(
        None, 
        description="URL to the generated audio file (if voice output was requested)"
    )


class VoiceInputRequest(BaseModel):
    """Request model for voice input"""
    audio_data: str = Field(
        ..., 
        description="Base64-encoded audio data"
    )
    return_audio: bool = Field(
        default=True, 
        description="Whether to return audio response"
    )
    request_id: str = Field(
        ..., 
        description="Unique identifier for the request"
    )


class VoiceOutputResponse(BaseModel):
    """Response model for voice output"""
    audio_data: str = Field(
        ..., 
        description="Base64-encoded audio data"
    )
    format: str = Field(
        default="mp3", 
        description="Format of the audio data"
    )


class StockData(BaseModel):
    """Model for stock data"""
    ticker: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    price: float = Field(..., description="Current price")
    change_percent: float = Field(..., description="Percentage change")
    volume: int = Field(..., description="Trading volume")
    
    @validator('change_percent')
    def round_change_percent(cls, v):
        """Round change percent to 2 decimal places"""
        return round(v, 2)


class EarningsData(BaseModel):
    """Model for earnings data"""
    ticker: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    eps_estimate: float = Field(..., description="EPS estimate")
    eps_actual: float = Field(..., description="Actual EPS")
    surprise_percent: float = Field(..., description="Surprise percentage")
    
    @validator('surprise_percent', 'eps_estimate', 'eps_actual')
    def round_values(cls, v):
        """Round values to 2 decimal places"""
        return round(v, 2)


class PortfolioAllocation(BaseModel):
    """Model for portfolio allocation data"""
    sector: str = Field(..., description="Sector name")
    region: str = Field(..., description="Geographic region")
    allocation_percent: float = Field(..., description="Allocation percentage")
    previous_percent: float = Field(..., description="Previous allocation percentage")
    change: float = Field(..., description="Change in allocation")
    
    @validator('allocation_percent', 'previous_percent', 'change')
    def round_percentages(cls, v):
        """Round percentages to 2 decimal places"""
        return round(v, 2)