from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx
from pydantic import BaseModel
import json
import logging
from typing import Dict, Any, Optional, List
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Finance Assistant Orchestrator")

# Base agent service URLs (update these with your actual service URLs when deployed)
SERVICE_URLS = {
    "voice": "http://voice-agent:8001",
    "api": "http://api-agent:8002",
    "scraping": "http://scraping-agent:8003",
    "retriever": "http://retriever-agent:8004",
    "analysis": "http://analysis-agent:8005",
    "language": "http://language-agent:8006"
}

# For local development
LOCAL_SERVICE_URLS = {
    "voice": "http://localhost:8001",
    "api": "http://localhost:8002",
    "scraping": "http://localhost:8003",
    "retriever": "http://localhost:8004",
    "analysis": "http://localhost:8005",
    "language": "http://localhost:8006"
}

# Use local URLs for development
ACTIVE_URLS = LOCAL_SERVICE_URLS

class QueryRequest(BaseModel):
    query: str
    use_voice: bool = False
    confidence_threshold: float = 0.7

class AgentResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    confidence: Optional[float] = None
    error: Optional[str] = None

async def call_agent(agent_name: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make an async call to an agent microservice."""
    try:
        url = f"{ACTIVE_URLS[agent_name]}/{endpoint}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Error calling {agent_name} agent: {str(e)}")
        return {"success": False, "error": f"Agent service error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error with {agent_name} agent: {str(e)}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

async def process_market_brief(query: str) -> Dict[str, Any]:
    """Process the market brief request through the agent pipeline."""
    # Step 1: Get market data from API agent
    market_data = await call_agent("api", "get_market_data", {"query": query})
    if not market_data.get("success", False):
        return market_data
    
    # Step 2: Get news and filings from scraping agent
    news_data = await call_agent("scraping", "get_news", {"query": query})
    if not news_data.get("success", False):
        return news_data
    
    # Step 3: Retrieve relevant past data using retriever agent
    retrieval_data = await call_agent("retriever", "retrieve", {
        "query": query,
        "market_data": market_data.get("data", {}),
        "news_data": news_data.get("data", {})
    })
    
    # Check confidence threshold
    confidence = retrieval_data.get("confidence", 0)
    if confidence < 0.7:  # Configurable threshold
        return {
            "success": False,
            "error": "Low confidence in retrieved data",
            "confidence": confidence,
            "request_clarification": True
        }
    
    # Step 4: Run analysis
    analysis_data = await call_agent("analysis", "analyze", {
        "query": query,
        "market_data": market_data.get("data", {}),
        "news_data": news_data.get("data", {}),
        "retrieval_data": retrieval_data.get("data", {})
    })
    
    # Step 5: Generate narrative with language agent
    narrative = await call_agent("language", "generate", {
        "query": query,
        "analysis": analysis_data.get("data", {}),
        "market_data": market_data.get("data", {}),
        "news_data": news_data.get("data", {})
    })
    
    return {
        "success": True,
        "data": {
            "narrative": narrative.get("data", {}).get("text", ""),
            "market_data": market_data.get("data", {}),
            "analysis": analysis_data.get("data", {}),
            "confidence": confidence
        }
    }

@app.post("/process")
async def process_query(request: QueryRequest):
    """Main endpoint to process user queries."""
    query = request.query
    
    # If voice input is enabled, convert speech to text first
    if request.use_voice:
        voice_response = await call_agent("voice", "speech_to_text", {"audio_file": "temp_audio.wav"})
        if not voice_response.get("success", False):
            raise HTTPException(status_code=400, detail="Failed to process voice input")
        query = voice_response.get("data", {}).get("text", query)
    
    # Process the query through the agent pipeline
    result = await process_market_brief(query)
    
    # If voice output is requested, convert text back to speech
    if request.use_voice and result.get("success", False):
        narrative = result.get("data", {}).get("narrative", "")
        voice_output = await call_agent("voice", "text_to_speech", {"text": narrative})
        if voice_output.get("success", False):
            result["data"]["audio_file"] = voice_output.get("data", {}).get("audio_file")
    
    # Handle low confidence case
    if not result.get("success", False) and result.get("request_clarification", False):
        return {
            "success": False, 
            "error": "Need more information", 
            "request_clarification": True
        }
    
    return result

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "Finance Assistant Orchestrator API",
        "version": "1.0.0",
        "endpoints": [
            "/process - Process a query through the agent pipeline",
            "/health - Health check endpoint"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)