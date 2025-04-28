import io
import torch
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from kokoro import KPipeline
import soundfile as sf
import logging

# Initialize FastAPI and logging
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# Serve static files (HTML/JS) from the static/ directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Global variable for the pipeline (initialized on first request)
pipeline = None

def initialize_model():
    """Initialize and quantize the TTS model"""
    global pipeline
    try:
        logger.info("Initializing Kokoro TTS pipeline...")
        pipeline = KPipeline(lang_code="a")
        
        # Quantize to reduce memory usage
        logger.info("Quantizing model...")
        pipeline.model = torch.quantization.quantize_dynamic(
            pipeline.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        logger.info("Model ready")
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="TTS engine failed to initialize"
        )

@app.get("/tts")
async def text_to_speech(
    text: str = Query(..., min_length=1, max_length=500, description="Text to synthesize")
):
    """Generate speech audio from text"""
    global pipeline
    
    try:
        # Lazy-load model on first request
        if pipeline is None:
            initialize_model()
        
        # Generate waveform (numpy array) at 24 kHz
        logger.info(f"Generating speech for: {text[:50]}...")
        waveform = pipeline(text)
        
        # Write to WAV in-memory
        buf = io.BytesIO()
        sf.write(buf, waveform, samplerate=24000, format="WAV")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="audio/wav")
    
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {str(e)}"
        )

# Health check endpoint for Cloud Run
@app.get("/health")
async def health_check():
    return {"status": "ready", "model_loaded": pipeline is not None}