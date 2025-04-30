# Dockerfile – Cloud Run container for Kokoro-TTS demo
FROM python:3.11-slim

# system libs for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# OPTIONAL: bake Kokoro weights to avoid HF download / 429
# RUN python - <<'PY'
# from huggingface_hub import snapshot_download
# snapshot_download('hexgrad/Kokoro-82M',
#   local_dir='/root/.cache/huggingface/hub/models--hexgrad--Kokoro-82M',
#   local_dir_use_symlinks=False)
# PY

# app source
COPY . .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
