import base64, io, os, secrets
from enum import Enum
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google.cloud import texttospeech
from huggingface_hub import login as hf_login
from kokoro import KPipeline
from loguru import logger

# -------------------------------------------------------------------- #
#  Global init
# -------------------------------------------------------------------- #
MODEL_REPO = "hexgrad/Kokoro-82M"
HF_TOKEN   = os.getenv("HF_TOKEN")
VALID_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k}

pipeline: Optional[KPipeline] = None
gcp_client: Optional[texttospeech.TextToSpeechClient] = None

app = FastAPI(title="Kokoro / GCP TTS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # in prod: narrow to your web domain
    allow_methods=["POST"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------- #
#  Auth dependency
# -------------------------------------------------------------------- #
def verify_key(authorization: str = Header(...)):
    try:
        scheme, token = authorization.split()
    except ValueError:
        raise HTTPException(401, "Malformed Authorization header")
    if scheme.lower() != "bearer" or token not in VALID_KEYS:
        raise HTTPException(401, "Invalid API key")

# -------------------------------------------------------------------- #
#  Pydantic I/O
# -------------------------------------------------------------------- #
class Engine(str, Enum):
    kokoro = "kokoro-82m"
    gcp    = "gcp-tts"

class SpeechReq(BaseModel):
    model: Engine = Engine.kokoro
    input: str    = Field(..., min_length=1, max_length=500)
    voice: str    = "af_heart"

class SpeechResp(BaseModel):
    audio_base64: str

# -------------------------------------------------------------------- #
#  Startup
# -------------------------------------------------------------------- #
@app.on_event("startup")
def load_models():
    global pipeline, gcp_client
    if HF_TOKEN:
        hf_login(token=HF_TOKEN.strip())

    logger.info("Loading Kokoro …")
    pipeline = KPipeline(repo_id=MODEL_REPO, lang_code="a")
    logger.info("Kokoro ready ✅")

    gcp_client = texttospeech.TextToSpeechClient()
    logger.info("GCP client ready ✅")

# -------------------------------------------------------------------- #
#  Helper functions
# -------------------------------------------------------------------- #
def kokoro_bytes(text: str, voice: str) -> bytes:
    audio_iter = pipeline(text, voice=voice)
    buf = io.BytesIO()
    with sf.SoundFile(buf, "w", 24_000, 1, "PCM_16") as wav:
        if hasattr(audio_iter, "shape"):
            wav.write(audio_iter.reshape(-1, 1))
        else:
            for chunk in audio_iter:
                samples = chunk.audio if hasattr(chunk, "audio") else chunk
                wav.write(np.asarray(samples, np.float32).reshape(-1, 1))
    return buf.getvalue()

def gcp_bytes(text: str) -> bytes:
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name="en-US-Standard-F"
    )
    cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000
    )
    resp = gcp_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=cfg
    )
    return resp.audio_content

# -------------------------------------------------------------------- #
#  OpenAI-style endpoint
# -------------------------------------------------------------------- #
@app.post("/v1/audio/speech",
          response_model=SpeechResp,
          dependencies=[Depends(verify_key)])
def speech(req: SpeechReq):
    if req.model == Engine.kokoro:
        wav = kokoro_bytes(req.input, req.voice)
    else:
        wav = gcp_bytes(req.input)

    return SpeechResp(audio_base64=base64.b64encode(wav).decode())
