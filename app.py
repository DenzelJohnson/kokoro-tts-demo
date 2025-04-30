"""
app.py – Cloud-Run demo for hexgrad/Kokoro-82M (Kokoro-TTS)

Routes
------
/               → static/index.html  (demo page)
/tts            → text-to-speech 24 kHz WAV stream
/health         → {"status": "alive"}   (uptime probe)

Notes
-----
• Requires an HF read token in the env var HF_TOKEN (mounted via Secret Manager)
• Default voice is **af_heart**; pass ?voice=<id> to use another.
"""

import io
import os
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import login as hf_login
from kokoro import KPipeline
from loguru import logger

# -----------------------------------------------------------------------------
#  Environment
# -----------------------------------------------------------------------------
MODEL_REPO = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
LANG_CODE  = os.getenv("KOKORO_LANG", "a")          # “a” = generic English
HF_TOKEN   = os.getenv("HF_TOKEN")                  # injected via Secret Manager

pipeline: Optional[KPipeline] = None

app = FastAPI(title="Kokoro-TTS Demo")

# -----------------------------------------------------------------------------
#  Static files & landing page
# -----------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("static/index.html")

# -----------------------------------------------------------------------------
#  Health check
# -----------------------------------------------------------------------------
@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "alive"}

# -----------------------------------------------------------------------------
#  Startup – load model once
# -----------------------------------------------------------------------------
@app.on_event("startup")
def initialize_model() -> None:
    global pipeline

    if HF_TOKEN:
        hf_login(token=HF_TOKEN.strip())            # strip newline if secret via echo

    try:
        logger.info(f"Loading Kokoro pipeline from {MODEL_REPO} …")
        pipeline = KPipeline(repo_id=MODEL_REPO, lang_code=LANG_CODE)
        logger.info("Kokoro pipeline ready ✅")
    except Exception as exc:
        logger.exception("Model initialization failed ❌")
        raise RuntimeError("TTS engine failed to initialize") from exc

# -----------------------------------------------------------------------------
#  TTS endpoint
# -----------------------------------------------------------------------------
@app.get("/tts", tags=["tts"])
def text_to_speech(
    text: str = Query(..., min_length=1, max_length=500,
                      description="Text to synthesise"),
    voice: str = Query("af_heart", description="Voice name (see Kokoro docs)")
) -> StreamingResponse:
    """
    Synthesise speech from `text` using the selected `voice`.
    Streams a 24 kHz mono WAV without holding the full waveform in RAM.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="TTS engine not ready")

    try:
        logger.debug(f"TTS request | voice={voice!r} | text={text!r}")
        audio_iter = pipeline(text, voice=voice)   # ndarray OR generator of chunks

        # --- write chunks progressively to keep memory low ---
        buf = io.BytesIO()
        with sf.SoundFile(buf, mode="w", samplerate=24_000,
                          channels=1, format="WAV", subtype="PCM_16") as wavfile:
            for chunk in audio_iter:
                chunk_arr = np.asarray(chunk, dtype=np.float32).reshape(-1, 1)
                wavfile.write(chunk_arr)

        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")

    except Exception as exc:
        logger.exception("TTS generation failed ❌")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
