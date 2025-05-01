"""
app.py – Cloud-Run demo: choose Kokoro (local) or Google Cloud TTS.

Endpoints
---------
/               → static/index.html  (demo)
/tts            → WAV stream (engine=[kokoro|gcp])
/health         → {"status":"alive"}
"""

import io
import os
from enum import Enum
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import texttospeech
from huggingface_hub import login as hf_login
from kokoro import KPipeline
from loguru import logger

# --------------------------------------------------------------------------- #
# ENV
# --------------------------------------------------------------------------- #
MODEL_REPO = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
LANG_CODE  = os.getenv("KOKORO_LANG", "a")
HF_TOKEN   = os.getenv("HF_TOKEN")           # injected via Secret Manager

pipeline: Optional[KPipeline] = None
tts_client: Optional[texttospeech.TextToSpeechClient] = None

app = FastAPI(title="Kokoro-&-GCP TTS Demo")

# --------------------------------------------------------------------------- #
# Static + health
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("static/index.html")

@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "alive"}

# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def init_models() -> None:
    global pipeline, tts_client

    if HF_TOKEN:
        hf_login(token=HF_TOKEN.strip())

    logger.info(f"Loading Kokoro pipeline from {MODEL_REPO} …")
    pipeline = KPipeline(repo_id=MODEL_REPO, lang_code=LANG_CODE)
    logger.info("Kokoro ready ✅")

    tts_client = texttospeech.TextToSpeechClient()
    logger.info("GCP Text-to-Speech client ready ✅")

# --------------------------------------------------------------------------- #
# Engines enum
# --------------------------------------------------------------------------- #
class Engine(str, Enum):
    kokoro = "kokoro"
    gcp    = "gcp"

# --------------------------------------------------------------------------- #
# TTS route
# --------------------------------------------------------------------------- #
@app.get("/tts", tags=["tts"])
def text_to_speech(
    text:   str    = Query(...,  min_length=1, max_length=500,
                           description="Text to synthesise"),
    engine: Engine = Query(Engine.kokoro, description="kokoro | gcp"),
    voice:  str    = Query("af_heart", description="Kokoro voice (ignored by GCP)")
) -> StreamingResponse:

    if pipeline is None or tts_client is None:
        raise HTTPException(500, "Engines not ready")

    if engine == Engine.kokoro:
        return _kokoro_tts(text, voice)
    else:
        return _gcp_tts(text)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _kokoro_tts(text: str, voice: str) -> StreamingResponse:
    audio_iter = pipeline(text, voice=voice)        # ndarray OR generator
    buf = io.BytesIO()
    with sf.SoundFile(buf,
                      mode="w",
                      samplerate=24_000,
                      channels=1,
                      format="WAV",        # ← ADD THIS
                      subtype="PCM_16") as wav:
        if hasattr(audio_iter, "shape"):            # ndarray
            wav.write(audio_iter.reshape(-1, 1))
        else:                                       # generator of Result / ndarray
            for chunk in audio_iter:
                samples = chunk.audio if hasattr(chunk, "audio") else chunk
                wav.write(np.asarray(samples, np.float32).reshape(-1, 1))
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


def _gcp_tts(text: str) -> StreamingResponse:
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-F",        # simple female US voice
    )
    audio_cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_cfg,
    )
    return StreamingResponse(io.BytesIO(response.audio_content),
                             media_type="audio/wav")
