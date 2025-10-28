import numpy as np
from openai import OpenAI
from scipy.io.wavfile import read, write
import io
import os
from dotenv import load_dotenv
import warnings

load_dotenv()
client = OpenAI()

def text_to_voice(text, voice=None, speed=1.0):
    """
    Convert text to speech using OpenAI TTS
    OPTIMIZED: Smaller chunks, faster processing
    
    Args:
        text: Text to convert
        voice: Voice model to use
        speed: Speech speed (0.25 to 4.0)
    """
    try:
        if not voice:
            voice = os.getenv("VOICE", "coral")

        # Use faster TTS model
        model = os.getenv("TEXT_TO_VOICE_MODEL", "tts-1")  # tts-1 is faster than tts-1-hd
        
        # Request speech with speed optimization
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            response_format="wav",
            input=text,
            speed=speed,  # Slightly faster playback
        )

        # Read the stream in larger chunks for efficiency
        audio_bytes = bytearray()
        for chunk in response.iter_bytes(chunk_size=16384):  # Larger chunks
            audio_bytes.extend(chunk)

        if len(audio_bytes) < 1000:
            raise ValueError("Received too little audio data")

        # Create buffer and read WAV
        buffer = io.BytesIO(bytes(audio_bytes))
        buffer.seek(0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            sample_rate, audio_data = read(buffer)

        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Audio data is empty or corrupted")

        # Fix corrupted WAV header
        fixed_buffer = io.BytesIO()
        write(fixed_buffer, sample_rate, audio_data)
        fixed_buffer.seek(0)
        sample_rate, audio_data = read(fixed_buffer)

        # Normalize to float32
        if audio_data.dtype != np.float32:
            max_val = np.iinfo(audio_data.dtype).max if np.issubdtype(audio_data.dtype, np.integer) else 1.0
            audio_data = audio_data.astype(np.float32) / max_val

        # OPTIMIZATION: Smaller chunks for faster initial playback
        chunk_duration = 0.5  # 500ms chunks instead of 1s
        samples_per_chunk = int(sample_rate * chunk_duration)
        
        audio_chunks = [
            audio_data[i:i + samples_per_chunk]
            for i in range(0, len(audio_data), samples_per_chunk)
        ]

        return {
            "sample_rate": sample_rate,
            "audio_chunks": audio_chunks
        }

    except Exception as e:
        print(f"Error during text-to-speech: {e}")
        raise


def text_to_voice_streaming(text, voice=None, chunk_size=50):
    """
    Convert text to speech in smaller chunks for streaming
    This allows TTS to start before complete AI response
    
    Args:
        text: Text to convert
        voice: Voice model
        chunk_size: Number of characters per TTS chunk
        
    Yields:
        Audio chunk dictionaries
    """
    try:
        if not voice:
            voice = os.getenv("VOICE", "coral")
        
        # Split text into sentence-like chunks
        chunks = []
        current_chunk = ""
        
        for char in text:
            current_chunk += char
            if char in '.!?' and len(current_chunk) >= chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Generate TTS for each chunk
        for chunk in chunks:
            if not chunk:
                continue
            
            audio_result = text_to_voice(chunk, voice, speed=1.0)
            yield audio_result
            
    except Exception as e:
        print(f"Error in streaming TTS: {e}")
        raise