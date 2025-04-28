FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Kokoro model (avoids HF rate limits at runtime)
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='a')"

# Copy application code
COPY . .

# Port for Cloud Run
ENV PORT 8080

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]