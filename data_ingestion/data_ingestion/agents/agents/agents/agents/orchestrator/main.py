import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Finance Assistant Orchestrator")

# Service URLs - use environment variables with sensible defaults
class ServiceConfig:
    API_AGENT_URL = os.getenv("API_AGENT_URL", "http://localhost:8001")
    SCRAPING_AGENT_URL = os.getenv("SCRAPING_AGENT_URL", "http://localhost:8002")
    RETRIEVER_AGENT_URL = os.getenv("RETRIEVER_AGENT_URL", "http://localhost:8003")
    ANALYSIS_AGENT_URL = os.getenv("ANALYSIS_AGENT_URL", "http://localhost:8004")
    LANGUAGE_AGENT_URL = os.getenv("LANGUAGE_AGENT_URL", "http://localhost:8005")
    VOICE_AGENT_URL = os.getenv("VOICE_AGENT_URL", "http://localhost:8006")

class QueryRequest(BaseModel):
    query: str
    response_format: Optional[str] = "text"  # "text" or "voice"

class QueryOrchestrator:
    @staticmethod
    async def call_service(url: str, payload: Dict):
        """
        Make an async HTTP call to a microservice
        
        Args:
            url (str): Service endpoint URL
            payload (Dict): Request payload
        
        Returns:
            Response from the service
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=response.text)
                return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Service call error: {str(e)}")

    @classmethod
    async def process_market_brief_query(cls, request: QueryRequest):
        """
        Process a market brief query by coordinating between microservices
        
        Args:
            request (QueryRequest): User query request
        
        Returns:
            Processed query response
        """
        try:
            # Step 1: Determine query context
            query_lower = request.query.lower()
            
            # Handle market brief scenario for Asia tech stocks
            if "risk exposure" in query_lower and ("asia" in query_lower or "tech" in query_lower):
                # Step 2: Retrieve contextual documents
                retrieval_response = await cls.call_service(
                    f"{ServiceConfig.RETRIEVER_AGENT_URL}/retrieve",
                    {
                        "query": request.query, 
                        "filters": {"region": "Asia", "sector": "Tech"}, 
                        "top_k": 5
                    }
                )
                context_docs = retrieval_response.get("results", [])
                
                # Step 3: Fetch market data
                market_data_response = await cls.call_service(
                    f"{ServiceConfig.API_AGENT_URL}/risk_exposure",
                    {"region": "Asia", "sector": "Tech"}
                )
                market_data = market_data_response.get("data", {})
                
                # Step 4: Scrape earnings data
                companies = ["TSMC", "Samsung"]
                earnings_data = []
                for company in companies:
                    try:
                        scraping_response = await cls.call_service(
                            f"{ServiceConfig.SCRAPING_AGENT_URL}/scrape",
                            {"company": company, "data_type": "earnings", "limit": 2}
                        )
                        earnings_data.extend(scraping_response.get("data", []))
                    except Exception as scrape_error:
                        print(f"Scraping failed for {company}: {scrape_error}")
                
                # Step 5: Analyze collected data
                analysis_response = await cls.call_service(
                    f"{ServiceConfig.ANALYSIS_AGENT_URL}/analyze",
                    {
                        "query": request.query,
                        "context": context_docs,
                        "market_data": market_data,
                        "earnings_data": earnings_data
                    }
                )
                analysis_result = analysis_response.get("analysis", {})
                
                # Step 6: Generate natural language response
                language_response = await cls.call_service(
                    f"{ServiceConfig.LANGUAGE_AGENT_URL}/generate",
                    {
                        "query": request.query,
                        "analysis": analysis_result,
                        "context": context_docs
                    }
                )
                text_response = language_response.get("text", "")
                
                # Step 7: Handle voice conversion if requested
                if request.response_format == "voice":
                    try:
                        # Return voice endpoint for further processing
                        tts_endpoint = f"{ServiceConfig.VOICE_AGENT_URL}/text_to_speech"
                        return {
                            "status": "success",
                            "text": text_response,
                            "voice_endpoint": tts_endpoint,
                            "voice_payload": {"text": text_response}
                        }
                    except Exception as voice_error:
                        return {
                            "status": "partial_success",
                            "text": text_response,
                            "message": f"Voice conversion failed: {str(voice_error)}"
                        }
                else:
                    return {
                        "status": "success",
                        "text": text_response
                    }
            else:
                # Generic handling for unsupported queries
                return {
                    "status": "error",
                    "message": "Currently, only risk exposure queries for Asia tech stocks are supported."
                }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

# API Endpoints
@app.post("/process_query")
async def process_query(request: QueryRequest):
    """
    Main endpoint for processing user queries
    """
    return await QueryOrchestrator.process_market_brief_query(request)

# Optional: Health check endpoint
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}

# Ensure project root is in Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Run the application
def start_server():
    """
    Start the Uvicorn server
    """
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )

if __name__ == "__main__":
    start_server()
    """
Main FastAPI application entry point
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.router import api_router
from config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="Morning Market Brief Assistant API",
    description="API for the Morning Market Brief Assistant",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint with basic service information"""
    return {
        "service": "Morning Market Brief Assistant API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)