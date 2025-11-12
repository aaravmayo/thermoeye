# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    GUNICORN_CMD_ARGS="--workers=2 --threads=4 --timeout=90"

WORKDIR /app

# System deps for OpenCV runtime (video codecs, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install gunicorn

# Copy app
COPY . .

# Make sure required dirs exist at runtime (uploads/data can be volumes)
RUN mkdir -p /app/uploads /app/data

EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
