"""
app.py ­– Cloud-Run demo for hexgrad/Kokoro-82M (Kokoro-TTS)

• `/`           → static/index.html  (textbox + Play button)
• `/tts`        → text-to-speech 24 kHz WAV stream
• `/health`     → {"status": "alive"}  (for uptime probes)
"""

import io
import os
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import login as hf_login
from kokoro import KPipeline
from loguru import logger

# --------------------------------------------------------------------------- #
#  Environment / constants
# --------------------------------------------------------------------------- #
MODEL_REPO = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
LANG_CODE  = os.getenv("KOKORO_LANG", "a")     # “a” = generic English voice
HF_TOKEN   = os.getenv("HF_TOKEN")             # injected via Secret Manager

pipeline: Optional[KPipeline] = None           # populated at startup
app = FastAPI(title="Kokoro-TTS Demo")

# --------------------------------------------------------------------------- #
#  Static files & landing page
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def index() -> FileResponse:                   # GET /
    """Serve the demo HTML page."""
    return FileResponse("static/index.html")

# --------------------------------------------------------------------------- #
#  Health check  (kept separate for uptime monitoring)
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "alive"}

# --------------------------------------------------------------------------- #
#  Startup – download / load the model once per container
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def initialize_model() -> None:
    global pipeline

    if HF_TOKEN:
        hf_login(token=HF_TOKEN.strip())       # remove accidental newline

    try:
        logger.info(f"Loading Kokoro pipeline from {MODEL_REPO} …")
        pipeline = KPipeline(repo_id=MODEL_REPO, lang_code=LANG_CODE)
        logger.info("Kokoro pipeline ready ✅")
    except Exception as exc:
        logger.exception("Model initialization failed ❌")
        raise RuntimeError("TTS engine failed to initialize") from exc

# --------------------------------------------------------------------------- #
#  TTS endpoint
# --------------------------------------------------------------------------- #
@app.get("/tts", tags=["tts"])
def text_to_speech(
    text: str = Query(..., min_length=1, max_length=500,
                      description="Text to synthesise")
) -> StreamingResponse:
    """
    Synthesize speech from `text` and return a 24 kHz mono WAV stream.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="TTS engine not ready")

    try:
        logger.debug(f"TTS request: {text!r}")
        wav = pipeline(text)                   # numpy float32, shape (1, N)
        buf = io.BytesIO()
        sf.write(buf, wav.T, samplerate=24_000, format="WAV")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")
    except Exception as exc:
        logger.exception("TTS generation failed ❌")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
