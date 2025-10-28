from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from contextlib import asynccontextmanager
import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from txt_to_txt import *
from txt_to_voice import *
from voice_to_txt import *
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=8)  # Increased workers

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Voice GPT API starting up...")
    yield
    logger.info("Voice GPT API shutting down...")
    executor.shutdown(wait=True)

app = FastAPI(
    title="Voice GPT",
    description="AI-powered voice conversation API",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class AudioResponse(BaseModel):
    sample_rate: int = Field(..., gt=0, description="Audio sample rate in Hz")
    audio_chunks: List[List[float]] = Field(..., description="List of audio chunks")

class AudioChunkRequest(BaseModel):
    voice: str = Field(default="coral", description="Voice model to use")
    audio_chunk: List[float] = Field(..., min_length=1, description="Audio samples as float32")
    stream: bool = Field(default=False, description="Enable streaming response")
    
    @field_validator('audio_chunk')
    @classmethod
    def validate_audio_chunk(cls, v):
        if len(v) == 0:
            raise ValueError("Audio chunk cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Audio chunk must contain only numeric values")
        return v

class TalkResponse(BaseModel):
    transcription: str
    ai_response: str
    audio_response: AudioResponse

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

async def run_in_threadpool(func, *args, **kwargs):
    """Run CPU-bound operations in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


@app.post("/talk-stream")
async def talk_stream(audio_chunk: AudioChunkRequest):
    """
    STREAMING VERSION: Returns audio chunks as they're generated
    This provides the fastest perceived response time
    NOTE: Requires txt_to_txt_stream function - see optimized_txt_to_txt.py
    """
    async def generate():
        try:
            # Step 1: Transcribe
            audio_data = np.array(audio_chunk.audio_chunk, dtype=np.float32)
            
            if len(audio_data) == 0 or np.max(np.abs(audio_data)) < 0.01:
                yield json.dumps({"error": "No audio signal detected"}) + "\n"
                return
            
            transcribed_text = await run_in_threadpool(transcribe_audio_chunk, audio_data)
            
            if not transcribed_text or transcribed_text.strip() == "":
                yield json.dumps({"error": "No speech detected"}) + "\n"
                return
            
            # Send transcription immediately
            yield json.dumps({
                "type": "transcription",
                "text": transcribed_text
            }) + "\n"
            
            logger.info(f"Transcription: {transcribed_text}")
            
            # Step 2: Get AI response (non-streaming for now)
            ai_response = await run_in_threadpool(txt_to_txt, transcribed_text)
            
            if not ai_response or ai_response.strip() == "":
                yield json.dumps({"error": "AI response generation failed"}) + "\n"
                return
            
            # Send AI response
            yield json.dumps({
                "type": "ai_text",
                "text": ai_response
            }) + "\n"
            
            logger.info(f"AI Response: {ai_response[:100]}...")
            
            # Step 3: Generate TTS
            audio_response = await run_in_threadpool(
                text_to_voice, 
                ai_response, 
                voice=audio_chunk.voice
            )
            
            if audio_response and 'audio_chunks' in audio_response:
                # Stream audio chunks
                for idx, audio_chunk_data in enumerate(audio_response['audio_chunks']):
                    chunk_list = audio_chunk_data.tolist() if isinstance(audio_chunk_data, np.ndarray) else audio_chunk_data
                    
                    yield json.dumps({
                        "type": "audio_chunk",
                        "sample_rate": audio_response['sample_rate'],
                        "chunk": chunk_list,
                        "index": idx,
                        "total": len(audio_response['audio_chunks'])
                    }) + "\n"
            
            # Send completion
            yield json.dumps({
                "type": "complete",
                "ai_response": ai_response
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield json.dumps({"error": str(e)}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/talk-parallel", response_model=TalkResponse)
async def talk_parallel(audio_chunk: AudioChunkRequest) -> dict:
    """
    PARALLEL VERSION: Processes AI response with optimized settings
    Best for non-streaming requests
    """
    try:
        audio_data = np.array(audio_chunk.audio_chunk, dtype=np.float32)
        
        if len(audio_data) == 0 or np.max(np.abs(audio_data)) < 0.01:
            raise HTTPException(status_code=400, detail="No audio signal detected.")
        
        # Step 1: Transcribe (must be sequential)
        transcribed_text = await run_in_threadpool(transcribe_audio_chunk, audio_data)
        
        if not transcribed_text or transcribed_text.strip() == "":
            raise HTTPException(status_code=400, detail="No speech detected.")
        
        logger.info(f"Transcription: {transcribed_text}")
        
        # Step 2: Get AI response with faster model
        ai_response = await run_in_threadpool(txt_to_txt, transcribed_text)
        
        if not ai_response or ai_response.strip() == "":
            raise HTTPException(status_code=500, detail="AI response generation failed.")
        
        logger.info(f"AI Response: {ai_response[:100]}...")
        
        # Step 3: TTS with optimized settings
        audio_response = await run_in_threadpool(
            text_to_voice, 
            ai_response, 
            voice=audio_chunk.voice
        )
        
        if not audio_response or 'audio_chunks' not in audio_response:
            raise HTTPException(status_code=500, detail="TTS failed.")
        
        audio_chunks_list = [
            chunk.tolist() if isinstance(chunk, np.ndarray) else chunk 
            for chunk in audio_response['audio_chunks']
        ]
        
        return {
            "transcription": transcribed_text,
            "ai_response": ai_response,
            "audio_response": {
                "sample_rate": audio_response['sample_rate'],
                "audio_chunks": audio_chunks_list
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/talk", response_model=TalkResponse)
async def talk_to_assistant(audio_chunk: AudioChunkRequest) -> dict:
    """Standard talk endpoint - calls optimized parallel version"""
    return await talk_parallel(audio_chunk)


@app.post("/talk-fast", response_model=TalkResponse)
async def talk_to_assistant_fast(audio_chunk: AudioChunkRequest) -> dict:
    """Fast talk endpoint - calls optimized parallel version"""
    return await talk_parallel(audio_chunk)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "Voice GPT Assistant"}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Voice GPT API"}

@app.get("/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": [
            "alloy", "echo", "fable", "onyx",
            "nova", "shimmer", "coral",
        ]
    }