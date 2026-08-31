# ── Dockerfile ────────────────────────────────────────────────
FROM python:3.10-slim

# System deps: FFmpeg + build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (CPU-only torch for HF Spaces)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces runs as non-root
RUN mkdir -p outputs temp cache && chmod 777 outputs temp cache

EXPOSE 7860

CMD ["python", "app.py"]