# Python STT Server with onnx_asr
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install CPU-only PyTorch and torchaudio first to avoid downloading CUDA binaries (~2-3GB)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create data directory for SQLite
RUN mkdir -p /data

# Environment variables
ENV MODEL_NAME=nemo-parakeet-tdt-0.6b-v3
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DB_PATH=/data/transcriptions.db
ENV PYTHONUNBUFFERED=1
# Disable ONNX Runtime external data path validation (required for HuggingFace cached models)
ENV ORT_DISABLE_EXTERNAL_INITIALIZERS_PATH_VALIDATION=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "main.py"]
