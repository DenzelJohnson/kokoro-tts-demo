# Use a slim Python base
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Copy & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Listen on port 8080 (Cloud Run default)
ENV PORT 8080

# Launch Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
