import io
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from kokoro import KPipeline
import soundfile as sf

app = FastAPI()

# Serve files from static/ at root
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Initialize Kokoro pipeline
pipeline = KPipeline(lang_code="a")
pipeline.model = torch.quantization.quantize_dynamic(
    pipeline.model,
    {torch.nn.Linear},               # quantize all Linear layers
    dtype=torch.qint8                 # use 8-bit ints
)
@app.get("/tts")
async def tts(text: str = Query(..., min_length=1)):
    # Generate waveform (numpy array) at 24 kHz
    waveform = pipeline(text)
    # Write to WAV in-memory
    buf = io.BytesIO()
    sf.write(buf, waveform, samplerate=24000, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")