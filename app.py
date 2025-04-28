import io
import torch
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from kokoro import KPipeline
import soundfile as sf
import logging
import os

# Initialize FastAPI
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# Initialize model variable
pipeline = None

# Serve static files using FileResponse for better compatibility
@app.get("/")
async def serve_index():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    raise HTTPException(status_code=404, detail="Index file not found")

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    static_file = os.path.join("static", file_path)
    if os.path.exists(static_file):
        return FileResponse(static_file)
    raise HTTPException(status_code=404, detail="File not found")

# TTS Endpoint
@app.get("/tts")
async def text_to_speech(
    text: str = Query(..., min_length=1, max_length=500)
):
    global pipeline
    try:
        if pipeline is None:
            logger.info("Initializing Kokoro TTS pipeline...")
            pipeline = KPipeline(lang_code="a")
            pipeline.model = torch.quantization.quantize_dynamic(
                pipeline.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        
        waveform = pipeline(text)
        buf = io.BytesIO()
        sf.write(buf, waveform, samplerate=24000, format="WAV")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")
    
    except Exception as e:
        logger.error(f"TTS Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}