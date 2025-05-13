"""
FastAPI router definitions for the API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from typing import Dict, Any, Optional, List

from api.models import (
    MarketBriefRequest,
    MarketBriefResponse,
    VoiceInputRequest,
    VoiceOutputResponse
)
from orchestrator.agent_manager import get_orchestrator

api_router = APIRouter(prefix="/api/v1")


@api_router.post("/market-brief", response_model=MarketBriefResponse)
async def get_market_brief(request: MarketBriefRequest):
    """
    Generate a market brief based on the provided question
    """
    try:
        orchestrator = get_orchestrator()
        response = await orchestrator.process_query(request.question)
        return MarketBriefResponse(
            brief=response["brief"],
            confidence=response["confidence"],
            sources=response["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating market brief: {str(e)}")


@api_router.post("/voice-input", response_model=MarketBriefResponse)
async def process_voice_input(request: VoiceInputRequest, background_tasks: BackgroundTasks):
    """
    Process voice input and return a market brief
    """
    try:
        orchestrator = get_orchestrator()
        # First convert speech to text
        text_query = await orchestrator.voice_agent.speech_to_text(request.audio_data)
        
        # Then process the query as normal
        response = await orchestrator.process_query(text_query)
        
        # Generate audio in the background if requested
        if request.return_audio:
            background_tasks.add_task(
                orchestrator.voice_agent.text_to_speech,
                response["brief"],
                request.request_id
            )
            audio_url = f"/api/v1/voice-output/{request.request_id}"
        else:
            audio_url = None
            
        return MarketBriefResponse(
            brief=response["brief"],
            confidence=response["confidence"],
            sources=response["sources"],
            audio_url=audio_url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice input: {str(e)}")


@api_router.get("/voice-output/{request_id}", response_model=VoiceOutputResponse)
async def get_voice_output(request_id: str):
    """
    Retrieve the generated voice output for a specific request
    """
    try:
        orchestrator = get_orchestrator()
        audio_data = await orchestrator.voice_agent.get_audio(request_id)
        if not audio_data:
            raise HTTPException(status_code=404, detail="Audio not found or still processing")
            
        return VoiceOutputResponse(
            audio_data=audio_data,
            format="mp3"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving voice output: {str(e)}")


@api_router.get("/health")
async def health_check():
    """
    Health check endpoint for API
    """
    return {"status": "healthy"}