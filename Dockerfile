# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support (for chatterbox-tts compatibility)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/test/cu128

# Copy requirements and install other dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-dotenv python-multipart requests psutil chatterbox-tts

# Copy application code
COPY app/ ./app/
COPY main.py ./

# Copy voice sample if it exists (optional, can be mounted)
COPY voice-sample.mp3 ./voice-sample.mp3

# Create directories for model cache and voice library (separate from source code)
RUN mkdir -p /cache /voices

# Set default environment variables
ENV PORT=4123
ENV EXAGGERATION=0.5
ENV CFG_WEIGHT=0.5
ENV TEMPERATURE=0.8
ENV VOICE_SAMPLE_PATH=/app/voice-sample.mp3
ENV MAX_CHUNK_LENGTH=280
ENV MAX_TOTAL_LENGTH=3000
ENV DEVICE=auto
ENV MODEL_CACHE_DIR=/cache
ENV VOICE_LIBRARY_DIR=/voices
ENV HOST=0.0.0.0

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with the new entry point
CMD ["python", "main.py"] 