FROM python:3.11-slim

WORKDIR /app

# Install system dependencies + git (required for HF downloads)
RUN apt-get update && \
    apt-get install -y libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

# Set up Hugging Face cache directory
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p ${HF_HOME}

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and explicitly cache all model files
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='hexgrad/Kokoro-82M', local_dir='/app/.cache/huggingface/hexgrad_Kokoro-82M')"

# Copy application code
COPY . .

# Set PORT for Cloud Run
ENV PORT 8080

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]