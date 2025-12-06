# Root-level Dockerfile to build the FastAPI backend on Render/Vercel.
# Build context is repository root; we only copy backend sources/deps.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: build tools, Tesseract OCR, and libpq for psycopg2
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (backend/requirements.txt lives in repo root/backend)
COPY backend/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy only the backend app into the image
COPY backend /app/backend

# Ensure Python can resolve the backend package
ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
