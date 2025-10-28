import io
import os
import numpy as np
from openai import OpenAI
from scipy.io.wavfile import write
from dotenv import load_dotenv
import logging

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)

# CRITICAL: Use the ACTUAL sample rate from the frontend
# Most browsers record at 48kHz or 44.1kHz, NOT 16kHz
def transcribe_audio_chunk(audio_chunk: np.ndarray, source_sample_rate: int = 48000) -> str:
    """
    Transcribe audio chunk using OpenAI Whisper
    
    Args:
        audio_chunk: numpy array of audio samples (float32, -1 to 1)
        source_sample_rate: Sample rate of the input audio (default 48000 for browsers)
        
    Returns:
        Transcribed text or None if error
    """
    try:
        if audio_chunk is None or len(audio_chunk) == 0:
            logger.warning("Empty audio chunk")
            return None
        
        # Handle stereo audio by converting to mono
        if len(audio_chunk.shape) > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
            logger.info(f"Converted stereo to mono")
        
        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Check audio stats
        max_val = np.max(np.abs(audio_chunk))
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        logger.info(f"Audio stats: length={len(audio_chunk)}, max={max_val:.4f}, rms={rms:.4f}, "
                   f"duration={len(audio_chunk)/source_sample_rate:.2f}s @ {source_sample_rate}Hz")
        
        # Normalize if needed (but preserve quiet audio)
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val
            logger.info(f"Normalized audio from max {max_val:.4f} to 1.0")
        elif max_val < 0.01:
            # Audio is very quiet - amplify it
            audio_chunk = audio_chunk * (0.3 / max_val)
            logger.info(f"Amplified quiet audio from {max_val:.4f} to 0.3")
        
        # Convert to int16 for WAV file
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        
        # Create WAV buffer with CORRECT sample rate
        buffer = io.BytesIO()
        write(buffer, source_sample_rate, audio_int16)
        buffer.seek(0)
        wav_bytes = buffer.read()
        
        logger.info(f"Created WAV: {len(wav_bytes)} bytes")
        
        # Get model from environment
        model = os.getenv("VOICE_TO_TEXT_MODEL", "whisper-1")
        
        # Determine response format based on model
        # gpt-4o-mini-transcribe-api-ev3 only supports 'json' or 'text'
        # whisper-1 supports 'verbose_json'
        if "gpt-4o-mini" in model or "transcribe" in model:
            response_format = "json"
        else:
            response_format = "verbose_json"
        
        logger.info(f"Using model: {model} with format: {response_format}")
        
        # Transcribe with optimized settings
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=model,
            response_format=response_format,
            language="en",
            temperature=0.0,
        )
        
        # Extract text based on response format
        if response_format == "verbose_json":
            text = transcription.text.strip()
            
            # Log segment details if available
            if hasattr(transcription, 'segments') and transcription.segments:
                logger.info(f"Segments: {len(transcription.segments)}")
                for i, seg in enumerate(transcription.segments):
                    logger.info(f"  Segment {i}: '{seg.text}' ({seg.start:.2f}s - {seg.end:.2f}s)")
        else:
            # Simple json format - just returns text
            text = transcription.text.strip() if hasattr(transcription, 'text') else str(transcription).strip()
        
        if text:
            logger.info(f"✅ Transcribed: '{text}'")
        else:
            logger.warning("⚠️ Empty transcription result")
        
        return text if text else None
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}", exc_info=True)
        return None