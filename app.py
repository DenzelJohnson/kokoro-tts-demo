# app.py – FastAPI wrapper around Kokoro-TTS (hexgrad/Kokoro-82M)

import io
import os
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from huggingface_hub import login as hf_login
from kokoro import KPipeline
from loguru import logger

APP_PORT   = int(os.getenv("PORT", 8080))
MODEL_REPO = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
LANG_CODE  = os.getenv("KOKORO_LANG", "a")
HF_TOKEN   = os.getenv("HF_TOKEN")         # injected by Cloud Run secret
pipeline: Optional[KPipeline] = None       # populated on startup

app = FastAPI(title="Kokoro-TTS demo")


# ---------- start-up ---------- #
@app.on_event("startup")
def initialize_model() -> None:
    """Download / load Kokoro model once at container start."""
    global pipeline

    if HF_TOKEN:
        hf_login(token=HF_TOKEN.strip())   # strip newline if secret was made with echo

    try:
        logger.info(f"Loading Kokoro pipeline from {MODEL_REPO} …")
        pipeline = KPipeline(repo_id=MODEL_REPO, lang_code=LANG_CODE)  # .no .eval()
        logger.info("Kokoro pipeline ready ✅")
    except Exception as exc:
        logger.exception("Model initialization failed")
        raise RuntimeError("TTS engine failed to initialize") from exc


# ---------- health ---------- #
@app.get("/", tags=["health"])
def root() -> dict:
    return {"status": "alive"}


# ---------- TTS ---------- #
@app.get("/tts", tags=["tts"])
def text_to_speech(
    text: str = Query(..., min_length=1, max_length=500, description="Text to synthesise")
) -> StreamingResponse:
    if pipeline is None:
        raise HTTPException(status_code=500, detail="TTS engine not ready")

    try:
        logger.debug(f"TTS request: {text!r}")
        wav = pipeline(text)                 # numpy float32 (1, n_samples)
        buf = io.BytesIO()
        sf.write(buf, wav.T, 24_000, format="WAV")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
