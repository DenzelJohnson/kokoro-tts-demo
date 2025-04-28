FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Kokoro model (optional - removes cold start delay)
# RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='a')"

# Copy application files
COPY static/ ./static/
COPY app.py .

# Set environment variables
ENV PORT 8080
ENV PYTHONUNBUFFERED True

# Expose the port
EXPOSE 8080

# Run with uvicorn (better for FastAPI ASGI apps)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]