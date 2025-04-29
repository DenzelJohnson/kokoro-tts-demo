import io
import torch
import logging
import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from kokoro import KPipeline
import soundfile as sf
from huggingface_hub import snapshot_download

# Initialize FastAPI
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# Global variable for the pipeline
pipeline = None

def initialize_model():
    """Initialize and quantize the TTS model from local cache"""
    global pipeline
    try:
        logger.info("Initializing Kokoro TTS pipeline...")
        
        # Path to our pre-downloaded model
        model_path = "/app/.cache/huggingface/hexgrad_Kokoro-82M"
        
        # Verify model files exist
        required_files = ["config.json", "pytorch_model.bin"]
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                raise FileNotFoundError(f"Missing model file: {file}")
        
        # Initialize pipeline with local model
        pipeline = KPipeline(lang_code="a", repo_id=model_path)
        
        # Quantize to reduce memory usage (optional)
        pipeline.model = torch.quantization.quantize_dynamic(
            pipeline.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        logger.info("Model successfully loaded from cache")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="TTS engine failed to initialize"
        )

@app.get("/tts")
async def text_to_speech(
    text: str = Query(..., min_length=1, max_length=500)
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
        logger.error(f"TTS generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {str(e)}"
        )

@app.get("/")
async def serve_index():
    """Serve the frontend interface"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    raise HTTPException(status_code=404, detail="Frontend not found")

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {
        "status": "ready", 
        "model_loaded": pipeline is not None,
        "cache_exists": os.path.exists("/app/.cache/huggingface/hexgrad_Kokoro-82M")
    }