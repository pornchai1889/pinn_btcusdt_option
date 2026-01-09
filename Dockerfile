# syntax=docker/dockerfile:1

# ==============================================================================
# STAGE 1: Base Image & Runtime Dependencies
# ==============================================================================
FROM python:3.10-slim AS builder

# Optimize Python for Docker Execution
# PYTHONDONTWRITEBYTECODE: Prevents .pyc file creation (saves space/time)
# PYTHONUNBUFFERED: Ensures logs are flushed directly to stdout for real-time monitoring
# PIP_NO_CACHE_DIR: Disables pip cache to reduce image size
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install System Utilities
# 'curl' is required for the HEALTHCHECK command later
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- OPTIMIZATION START ---

# 1. Install PyTorch CPU-Only Version (The Critical Space Saver)
# We install this explicitly using the PyTorch CPU wheel index.
# This prevents installing NVIDIA CUDA drivers (~3GB) which are useless for this API.
RUN pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Install Remaining Inference Dependencies
COPY requirements.inference.txt .
RUN pip install -r requirements.inference.txt

# --- OPTIMIZATION END ---

# ==============================================================================
# STAGE 2: Model Artifacts Injection
# ==============================================================================
# Copy pre-trained model weights from the host machine.
# Ensure these paths match your local 'models' folder structure.

# 1. Inject CALL Option Model
COPY models/call/ft_2025-12-22_20-31-36 \
     /app/models/call

# 2. Inject PUT Option Model
COPY models/put/ft_2025-12-25_14-28-13 \
     /app/models/put

# ==============================================================================
# STAGE 3: Application Assembly & Security
# ==============================================================================
# Copy Source Code
COPY src/ /app/src/

# Create a non-root user (Security Best Practice)
# Running as root inside a container is a security vulnerability.
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Define Default Runtime Configuration
ENV CALL_MODEL_DIR=/app/models/call \
    PUT_MODEL_DIR=/app/models/put \
    DEVICE=cpu \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

# Expose API Port
EXPOSE 8000

# Container Healthcheck (Liveness Probe)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/v1/health || exit 1

# Start the Application Server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]