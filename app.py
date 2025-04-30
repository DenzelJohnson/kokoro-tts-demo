# app.py – FastAPI wrapper around Kokoro-TTS
import io
import os
import logging
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from huggingface_hub import login as hf_login
from kokoro import KPipeline
from loguru import logger

APP_PORT: int = int(os.getenv("PORT", 8080))
MODEL_REPO: str = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
HF_TOKEN:   Optional[str] = os.getenv("HF_TOKEN")   # optional, but avoids 429s
LANG_CODE:  str = os.getenv("KOKORO_LANG", "a")     # “a” = generic English voice

app = FastAPI(title="Kokoro-TTS demo")
pipeline: Optional[KPipeline] = None     # will hold the loaded model


# ---------- start-up & model init ---------- #
@app.on_event("startup")
def initialize_model() -> None:
    """
    Load Kokoro model once at container start-up.
    Raise RuntimeError if anything is missing – that will fail the
    Cloud Run revision immediately instead of giving first user a 500.
    """
    global pipeline

    if HF_TOKEN:
        logger.info("Logging into Hugging Face Hub with token (HF_TOKEN)")
        hf_login(token=HF_TOKEN)

    try:
        logger.info(f"Loading Kokoro pipeline from {MODEL_REPO} …")
        pipeline = KPipeline(repo_id=MODEL_REPO, lang_code=LANG_CODE).eval()
        logger.info("Kokoro pipeline ready ✅")
    except Exception as exc:
        logger.exception("Model initialization failed ❌")
        raise RuntimeError("TTS engine failed to initialize") from exc


# ---------- health endpoint ---------- #
@app.get("/", tags=["health"])
def root() -> dict:
    return {"status": "alive"}


# ---------- TTS endpoint ---------- #
@app.get("/tts", tags=["tts"])
def text_to_speech(
    text: str = Query(..., min_length=1, max_length=500, description="Text to synthesise"),
) -> StreamingResponse:
    """
    Convert text → speech (24 kHz mono WAV).
    Returns audio/wav stream; client receives immediately after synthesis.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="TTS engine not ready")

    try:
        logger.debug(f"TTS request: {text!r}")
        waveform = pipeline(text)                # NumPy float32, shape (1, n_samples)
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, waveform.T, samplerate=24_000, format="WAV")
        wav_bytes.seek(0)
        return StreamingResponse(wav_bytes, media_type="audio/wav")
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
