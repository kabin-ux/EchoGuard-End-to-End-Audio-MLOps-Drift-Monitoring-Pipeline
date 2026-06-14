"""
app/main.py
───────────
FastAPI application — audio genre classification with drift monitoring.

Endpoints
─────────
  POST /predict   – Upload audio file → genre label + drift report
  GET  /metrics   – Prometheus metrics endpoint (scraped every 5 s)
  GET  /health    – Liveness probe for Docker / k8s health checks
  GET  /          – Human-readable API info

Prometheus Metrics
──────────────────
  audio_predictions_total        Counter   Labels: genre, drift_detected
  prediction_latency_seconds     Histogram Prediction wall-clock time
  feature_drift_p_value          Gauge     Minimum p-value across features (worst-case)
  audio_upload_bytes             Histogram Distribution of uploaded file sizes
  drift_window_size              Gauge     Current rolling window depth
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

from app.utils.audio import extract_audio_features
from app.utils.drift import evaluate_drift, get_window_size

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Artifact paths ────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

# ── Prometheus metrics ────────────────────────────────────────────────────────
audio_predictions_total = Counter(
    name="audio_predictions_total",
    documentation="Total number of genre predictions served.",
    labelnames=["genre", "drift_detected"],
)

prediction_latency_seconds = Histogram(
    name="prediction_latency_seconds",
    documentation="End-to-end prediction latency in seconds.",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

feature_drift_p_value = Gauge(
    name="feature_drift_p_value",
    documentation=(
        "Minimum KS p-value across all features in the current window. "
        "Low values indicate potential data drift."
    ),
)

audio_upload_bytes = Histogram(
    name="audio_upload_bytes",
    documentation="Size distribution of uploaded audio files in bytes.",
    buckets=[1_024, 10_240, 102_400, 512_000, 1_048_576, 5_242_880, 10_485_760],
)

drift_window_size_gauge = Gauge(
    name="drift_window_size",
    documentation="Current number of samples in the drift detection rolling window.",
)

# ── Model loader (lazy-loaded on first request) ───────────────────────────────
_model_bundle: dict | None = None


def _load_model() -> dict:
    """Load and cache the model bundle from disk."""
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. "
                "Run `python scripts/generate_artifacts.py` first."
            )
        logger.info("Loading model bundle from %s …", MODEL_PATH)
        _model_bundle = joblib.load(MODEL_PATH)
        genres = _model_bundle.get("genres", [])
        logger.info("Model loaded. Supports genres: %s", genres)
    return _model_bundle


# ── Application lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-up: pre-load model on startup to reduce first-request latency."""
    logger.info("Application startup: pre-loading model …")
    try:
        _load_model()
        logger.info("Model pre-loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning("Model not found during startup: %s", exc)
        logger.warning("The /predict endpoint will return 503 until artifacts are generated.")
    yield
    logger.info("Application shutdown.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Audio Genre Classification API",
    description=(
        "Real-time audio genre classification with Kolmogorov-Smirnov "
        "data drift monitoring and Prometheus observability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=JSONResponse, tags=["Info"])
async def root() -> dict[str, Any]:
    """Return API metadata and endpoint listing."""
    return {
        "service": "Audio Genre Classification API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload an audio file and receive a genre prediction.",
            "GET  /metrics": "Prometheus metrics scrape endpoint.",
            "GET  /health":  "Liveness probe.",
            "GET  /docs":    "Interactive Swagger UI.",
        },
        "supported_formats": ["wav", "mp3", "flac", "ogg", "aiff"],
    }


@app.get("/health", tags=["Ops"])
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 OK when the service is ready."""
    model_status = "loaded" if _model_bundle is not None else "not_loaded"
    return {"status": "ok", "model": model_status}


@app.post("/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(
        ...,
        description="Audio file to classify (WAV, MP3, FLAC, OGG).",
    )
) -> JSONResponse:
    """
    Classify the genre of an uploaded audio file.

    The endpoint:
    1. Reads the uploaded bytes.
    2. Extracts acoustic features with Librosa.
    3. Runs the Random Forest classifier.
    4. Evaluates KS-based data drift against the baseline distribution.
    5. Records Prometheus metrics.

    Returns a JSON object containing:
    - ``genre``       – Predicted genre label.
    - ``confidence``  – Max class probability from the RF.
    - ``drift``       – Drift report (p-values, drifted features, overall flag).
    - ``latency_ms``  – End-to-end processing time in milliseconds.
    """
    t_start = time.perf_counter()

    # ── 1. Read uploaded file ─────────────────────────────────────────────────
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    audio_upload_bytes.observe(len(file_bytes))
    logger.info(
        "Received audio file: name=%s  size=%d bytes  content_type=%s",
        file.filename,
        len(file_bytes),
        file.content_type,
    )

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    try:
        features = extract_audio_features(file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio processing failed: {exc}",
        )
    except RuntimeError as exc:
        logger.exception("Feature extraction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction error: {exc}",
        )

    feature_vector: list[float] = features["feature_vector"]

    # ── 3. Model inference ────────────────────────────────────────────────────
    try:
        bundle = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    clf = bundle["model"]
    le: Any = bundle["label_encoder"]

    X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    pred_idx = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    genre: str = le.inverse_transform([pred_idx])[0]
    confidence: float = float(proba.max())

    logger.info("Prediction: genre=%s  confidence=%.3f", genre, confidence)

    # ── 4. Drift detection ────────────────────────────────────────────────────
    try:
        drift_report = evaluate_drift(feature_vector)
    except Exception as exc:
        logger.warning("Drift evaluation failed (non-fatal): %s", exc)
        drift_report = None

    # ── 5. Prometheus instrumentation ─────────────────────────────────────────
    t_elapsed = time.perf_counter() - t_start
    drift_str = "true" if (drift_report and drift_report.drift_detected) else "false"

    prediction_latency_seconds.observe(t_elapsed)
    audio_predictions_total.labels(genre=genre, drift_detected=drift_str).inc()
    drift_window_size_gauge.set(get_window_size())

    if drift_report:
        feature_drift_p_value.set(drift_report.min_p_value)

    # ── 6. Build response ─────────────────────────────────────────────────────
    drift_payload: dict[str, Any] = {}
    if drift_report:
        drift_payload = {
            "drift_detected": drift_report.drift_detected,
            "drift_fraction": round(drift_report.drift_fraction, 4),
            "min_p_value": round(drift_report.min_p_value, 6),
            "drifted_features": drift_report.drifted_features,
            "window_size": drift_report.window_size,
            "p_values": {k: round(v, 6) for k, v in drift_report.p_values.items()},
        }
    else:
        drift_payload = {"drift_detected": False, "error": "Drift evaluation unavailable."}

    response_payload = {
        "genre": genre,
        "confidence": round(confidence, 4),
        "latency_ms": round(t_elapsed * 1000, 2),
        "drift": drift_payload,
        "features": {
            "mfcc_mean": [round(v, 4) for v in features["mfcc_mean"]],
            "spectral_centroid_mean": round(features["spectral_centroid_mean"], 2),
            "chroma_mean": round(features["chroma_mean"], 4),
        },
    }

    logger.info(
        "Request complete: genre=%s  latency=%.1f ms  drift=%s",
        genre, t_elapsed * 1000, drift_str,
    )
    return JSONResponse(content=response_payload, status_code=200)


@app.get("/metrics", tags=["Ops"])
async def metrics() -> PlainTextResponse:
    """
    Expose Prometheus metrics in the standard text exposition format.

    Prometheus should be configured to scrape this endpoint.
    The Content-Type is set to the standard ``text/plain; version=0.0.4``
    to ensure compatibility with all Prometheus versions.
    """
    data = generate_latest(REGISTRY)
    return PlainTextResponse(
        content=data.decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
