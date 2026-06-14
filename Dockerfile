# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Audio Genre Classification API
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build: dependencies installed in builder, only runtime files
# copied to the final slim image to minimise attack surface and image size.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed by librosa / soundfile (libsndfile) and audioread
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libsndfile1-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="mlops-team"
LABEL description="Audio Genre Classification API with drift monitoring"

RUN groupadd --system appgroup && useradd --system --gid appgroup appuser

WORKDIR /api

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/

# REMOVE the RUN python script line. 
# Instead, copy the compiled artifacts you built on your host machine:
COPY app/artifacts/ ./app/artifacts/

# Ensure the appuser has full ownership of the app directory
RUN chown -R appuser:appgroup /api

USER appuser

# ── Environment defaults ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/api \
    NUMBA_CACHE_DIR=/tmp

EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--access-log"]
