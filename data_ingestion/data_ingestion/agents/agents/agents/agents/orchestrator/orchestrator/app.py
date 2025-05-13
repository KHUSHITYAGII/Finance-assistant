from fastapi import FastAPI, HTTPException, Request, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import httpx
import logging
import os
import json
import time
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Finance Assistant Orchestrator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define service endpoints (should match docker-compose or local setup)
API_AGENT_URL = os.getenv("API_AGENT_URL", "http://api_agent:8001")
SCRAPING_AGENT_URL = os.getenv("SCRAPING_AGENT_URL", "http://scraping_agent:8002")
RETRIEVER_AGENT_URL = os.getenv("RETRIEVER_AGENT_URL", "http://retriever_agent:8003")
ANALYSIS_AGENT_URL = os.getenv("ANALYSIS_AGENT_URL", "http://analysis_agent:8004")
LANGUAGE_AGENT_URL = os.getenv("LANGUAGE_AGENT_URL", "http://language_agent:8005")
VOICE_AGENT_URL = os.getenv("VOICE_AGENT_URL", "http://voice_agent:8006")

# Request models
class QueryRequest(BaseModel):
    query: str
    response_format: str = "Text"  # "Text" or "Voice"

# Helper functions for agent communication
async def call_agent(url, data):
    """Call an agent service with provided data"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{url}/process", json=data)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Error calling agent at {url}: {e}")
        return {"error": f"Failed to reach agent service: {str(e)}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from agent at {url}: {e}")
        return {"error": f"Agent service error: {e.response.text}"}

@app.get("/")
async def root():
    return {"message": "Finance Assistant Orchestrator API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/process_query")
async def process_query(request: QueryRequest):
    """Process a finance query through the agent pipeline"""
    query = request.query
    response_format = request.response_format
    logger.info(f"Processing query: {query}, format: {response_format}")
    
    # Step 1: Call API Agent for financial data
    logger.info("Calling API Agent...")
    api_data = await call_agent(API_AGENT_URL, {"query": query})
    if "error" in api_data:
        logger.error(f"API Agent error: {api_data['error']}")
    
    # Step 2: Call Scraping Agent for web data
    logger.info("Calling Scraping Agent...")
    scraper_data = await call_agent(SCRAPING_AGENT_URL, {"query": query})
    if "error" in scraper_data:
        logger.error(f"Scraping Agent error: {scraper_data['error']}")
    
    # Step 3: Call Retriever Agent for vector search
    logger.info("Calling Retriever Agent...")
    retriever_data = await call_agent(RETRIEVER_AGENT_URL, {"query": query})
    if "error" in retriever_data:
        logger.error(f"Retriever Agent error: {retriever_data['error']}")
    
    # Step 4: Call Analysis Agent to process all data
    logger.info("Calling Analysis Agent...")
    analysis_data = await call_agent(
        ANALYSIS_AGENT_URL, 
        {
            "query": query,
            "api_data": api_data,
            "scraper_data": scraper_data,
            "retriever_data": retriever_data
        }
    )
    if "error" in analysis_data:
        logger.error(f"Analysis Agent error: {analysis_data['error']}")
    
    # Step 5: Call Language Agent for text response
    logger.info("Calling Language Agent...")
    language_data = await call_agent(
        LANGUAGE_AGENT_URL,
        {
            "query": query,
            "analysis_results": analysis_data
        }
    )
    if "error" in language_data:
        logger.error(f"Language Agent error: {language_data['error']}")
    
    # Process response based on format
    text_response = language_data.get("market_brief", "No market brief generated.")
    
    # Step 6: Call Voice Agent if voice response requested
    if response_format.lower() == "voice":
        logger.info("Calling Voice Agent for TTS...")
        voice_data = await call_agent(
            VOICE_AGENT_URL,
            {
                "query": query,
                "response_text": text_response
            }
        )
        if "error" in voice_data:
            logger.error(f"Voice Agent error: {voice_data['error']}")
            
        return {
            "text_response": text_response,
            "audio_path": voice_data.get("audio_file_path"),
            "format": "voice"
        }
    else:
        return {
            "text_response": text_response,
            "format": "text"
        }

# Run with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)