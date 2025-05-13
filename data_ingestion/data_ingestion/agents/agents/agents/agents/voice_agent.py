import os
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import speech_recognition as sr
import pyttsx3
import logging
import json
import threading
import whisper
import numpy as np
from io import BytesIO
import soundfile as sf
import asyncio
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Finance Voice Agent")

# Load Whisper model
model = whisper.load_model("base")

class SpeechToTextRequest(BaseModel):
    audio_file: str

class TextToSpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = 1.0

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    try:
        # Transcribe audio
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None

def text_to_speech(text, output_path, voice=None, speed=1.0):
    """Convert text to speech using pyttsx3."""
    try:
        engine = pyttsx3.init()
        
        # Set voice if provided
        if voice:
            voices = engine.getProperty('voices')
            for v in voices:
                if voice.lower() in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
        
        # Set speech rate
        engine.setProperty('rate', engine.getProperty('rate') * speed)
        
        # Generate speech
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        return True
    except Exception as e:
        logger.error(f"Error converting text to speech: {e}")
        return False

@app.post("/speech_to_text")
async def speech_to_text(file: UploadFile = File(...)):
    """Convert speech to text."""
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        
        # Write uploaded file to temporary file
        with open(temp_path, 'wb') as f:
            f.write(await file.read())
        
        # Transcribe audio
        text = transcribe_audio(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        if text:
            return {
                "success": True,
                "data": {
                    "text": text
                }
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")
    
    except Exception as e:
        logger.error(f"Error in speech_to_text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process speech: {str(e)}")

@app.post("/text_to_speech")
async def text_to_speech_endpoint(request: TextToSpeechRequest):
    """Convert text to speech."""
    try:
        # Create temporary file for output audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        output_path = temp_file.name
        
        # Generate speech
        success = text_to_speech(
            request.text, 
            output_path, 
            voice=request.voice, 
            speed=request.speed
        )
        
        if success:
            # Return the audio file
            return FileResponse(
                output_path, 
                media_type="audio/wav", 
                filename="speech.wav",
                headers={"Content-Disposition": "attachment"}
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to generate speech")
    
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

# Optional route for testing
@app.get("/test_tts")
async def test_tts():
    """Test text-to-speech functionality."""
    test_text = "Today, your Asia tech allocation is 22% of AUM, up from 18% yesterday. TSMC beat estimates by 4%, Samsung missed by 2%. Regional sentiment is neutral with a cautionary tilt due to rising yields."
    
    request = TextToSpeechRequest(text=test_text)
    return await text_to_speech_endpoint(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)