FROM python:3.11-slim

WORKDIR /app

# Install system deps + git (required for HF downloads)
RUN apt-get update && \
    apt-get install -y libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

# Set up Python and Hugging Face cache
ENV PIP_NO_CACHE_DIR=false \
    HF_HOME=/app/.cache/huggingface

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Kokoro model (avoid runtime downloads)
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='a')"

# Copy app code
COPY . .

# Set PORT for Cloud Run
ENV PORT 8080

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]