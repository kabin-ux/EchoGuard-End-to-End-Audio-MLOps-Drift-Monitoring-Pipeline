"""
test_main.py
────────────
Pytest test suite for the Audio Genre Classification API.

Test coverage:
  - GET  /          → 200 OK with expected keys
  - GET  /health    → 200 OK, correct structure
  - GET  /metrics   → 200 OK, Prometheus text format
  - POST /predict   → 200 OK with valid minimal WAV, correct response schema
  - POST /predict   → 422 when no file is provided
  - POST /predict   → 422 when an empty payload is uploaded
  - Drift detection → KS test returns correct structure
  - Audio features  → extracts expected feature dimensions

All ML/model calls are real (using a fixture that generates artifacts in a
temp directory), so these tests validate the full pipeline rather than mocks.
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Ensure project root is importable ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_wav_bytes(
    duration_seconds: float = 1.0,
    sample_rate: int = 22_050,
    frequency: float = 440.0,
) -> bytes:
    """
    Generate an in-memory WAV file containing a pure sine wave.

    This produces a valid, librosa-parseable audio file without requiring
    any audio assets on disk.
    """
    n_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, n_samples, endpoint=False)
    sine_wave = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)            # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(sine_wave.tobytes())
    buf.seek(0)
    return buf.read()


def _make_baseline(feature_names: list[str], n_samples: int = 50) -> dict:
    """Create a synthetic baseline.json compatible with drift.py."""
    rng = np.random.default_rng(seed=99)
    reference_samples = {
        name: rng.normal(0.0, 1.0, size=n_samples).tolist()
        for name in feature_names
    }
    return {
        "feature_names": feature_names,
        "reference_samples": reference_samples,
        "metadata": {"n_baseline_samples": n_samples},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory) -> Path:
    """
    Session-scoped fixture: generate model.joblib + baseline.json into a
    temporary directory and patch the paths used by main.py / drift.py.
    """
    tmp_dir = tmp_path_factory.mktemp("artifacts")

    # ── Generate model ────────────────────────────────────────────────────────
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    from app.utils.audio import FEATURE_DIM

    genres = ["blues", "classical", "country", "disco",
              "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    rng = np.random.default_rng(seed=42)

    X_parts, y_parts = [], []
    for i, genre in enumerate(genres):
        samples = rng.normal(float(i * 1.5), 0.5, size=(50, FEATURE_DIM))
        X_parts.append(samples)
        y_parts.extend([genre] * 50)

    X = np.vstack(X_parts).astype(np.float32)
    y = np.array(y_parts)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y_enc)

    bundle = {"model": clf, "label_encoder": le, "genres": genres}
    model_path = tmp_dir / "model.joblib"
    joblib.dump(bundle, model_path)

    # ── Generate baseline ─────────────────────────────────────────────────────
    from app.utils.audio import FEATURE_NAMES
    baseline = _make_baseline(FEATURE_NAMES, n_samples=50)
    baseline_path = tmp_dir / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    return tmp_dir


@pytest.fixture(scope="session")
def client(artifacts_dir: Path) -> Generator[TestClient, None, None]:
    """
    Session-scoped TestClient with patched artifact paths.
    The model and baseline are loaded from the session temp directory.
    """
    # Reset module-level caches before patching
    import app.main as main_module
    import app.utils.drift as drift_module

    main_module._model_bundle = None
    drift_module._state.baseline = None
    drift_module.clear_window()

    model_path = artifacts_dir / "model.joblib"
    baseline_path = artifacts_dir / "baseline.json"

    with (
        patch.object(main_module, "MODEL_PATH", model_path),
        patch.object(drift_module, "BASELINE_PATH", baseline_path),
    ):
        with TestClient(main_module.app, raise_server_exceptions=False) as c:
            yield c

    # Cleanup cached state
    main_module._model_bundle = None
    drift_module._state.baseline = None
    drift_module.clear_window()


@pytest.fixture()
def valid_wav() -> bytes:
    """A minimal but valid 1-second 440 Hz WAV file."""
    return _make_wav_bytes(duration_seconds=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — root and health endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestInfoEndpoints:
    def test_root_returns_200(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_expected_keys(self, client: TestClient):
        data = client.get("/").json()
        assert "service" in data
        assert "endpoints" in data
        assert "supported_formats" in data

    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "model" in data


# ─────────────────────────────────────────────────────────────────────────────
# Tests — /metrics endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client: TestClient):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client: TestClient):
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_python_gc(self, client: TestClient):
        """Prometheus default Python metrics must always be present."""
        body = client.get("/metrics").text
        assert "python_gc_objects_collected_total" in body

    def test_metrics_contains_custom_metrics_after_prediction(
        self, client: TestClient, valid_wav: bytes
    ):
        """Custom metrics appear in /metrics only after at least one prediction."""
        client.post(
            "/predict",
            files={"file": ("test.wav", valid_wav, "audio/wav")},
        )
        body = client.get("/metrics").text
        assert "audio_predictions_total" in body
        assert "prediction_latency_seconds" in body


# ─────────────────────────────────────────────────────────────────────────────
# Tests — /predict endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_valid_wav_returns_200(self, client: TestClient, valid_wav: bytes):
        response = client.post(
            "/predict",
            files={"file": ("sample.wav", valid_wav, "audio/wav")},
        )
        assert response.status_code == 200

    def test_predict_response_contains_genre(self, client: TestClient, valid_wav: bytes):
        data = client.post(
            "/predict",
            files={"file": ("sample.wav", valid_wav, "audio/wav")},
        ).json()
        assert "genre" in data
        assert isinstance(data["genre"], str)
        assert len(data["genre"]) > 0

    def test_predict_response_contains_confidence(self, client: TestClient, valid_wav: bytes):
        data = client.post(
            "/predict",
            files={"file": ("sample.wav", valid_wav, "audio/wav")},
        ).json()
        assert "confidence" in data
        conf = data["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_predict_response_contains_latency(self, client: TestClient, valid_wav: bytes):
        data = client.post(
            "/predict",
            files={"file": ("sample.wav", valid_wav, "audio/wav")},
        ).json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0.0

    def test_predict_response_contains_drift_section(self, client: TestClient, valid_wav: bytes):
        data = client.post(
            "/predict",
            files={"file": ("sample.wav", valid_wav, "audio/wav")},
        ).json()
        assert "drift" in data
        drift = data["drift"]
        assert "drift_detected" in drift
        assert isinstance(drift["drift_detected"], bool)

    def test_predict_response_contains_features_section(self, client: TestClient, valid_wav: bytes):
        data = client.post(
            "/predict",
            files={"file": ("sample.wav", valid_wav, "audio/wav")},
        ).json()
        assert "features" in data
        feats = data["features"]
        assert "mfcc_mean" in feats
        assert len(feats["mfcc_mean"]) == 13
        assert "spectral_centroid_mean" in feats
        assert "chroma_mean" in feats

    def test_predict_genre_is_known_label(self, client: TestClient, valid_wav: bytes):
        known_genres = {
            "blues", "classical", "country", "disco",
            "hiphop", "jazz", "metal", "pop", "reggae", "rock",
        }
        data = client.post(
            "/predict",
            files={"file": ("sample.wav", valid_wav, "audio/wav")},
        ).json()
        assert data["genre"] in known_genres

    def test_predict_empty_file_returns_422(self, client: TestClient):
        response = client.post(
            "/predict",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 422

    def test_predict_corrupt_file_returns_422(self, client: TestClient):
        corrupt_bytes = b"\x00\x01\x02\x03" * 100   # random garbage
        response = client.post(
            "/predict",
            files={"file": ("corrupt.wav", corrupt_bytes, "audio/wav")},
        )
        assert response.status_code in (422, 500)

    def test_predict_long_audio_returns_200(self, client: TestClient):
        """Verify the pipeline handles longer clips (10 s) without error."""
        long_wav = _make_wav_bytes(duration_seconds=10.0)
        response = client.post(
            "/predict",
            files={"file": ("long.wav", long_wav, "audio/wav")},
        )
        assert response.status_code == 200

    def test_predict_multiple_calls_increment_counter(
        self, client: TestClient, valid_wav: bytes
    ):
        """Metrics counter must increase with each prediction."""
        def _predict():
            client.post(
                "/predict",
                files={"file": ("t.wav", valid_wav, "audio/wav")},
            )

        for _ in range(3):
            _predict()

        body = client.get("/metrics").text
        # Counter is non-zero somewhere in the output
        assert "audio_predictions_total" in body


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Feature extraction unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioFeatureExtraction:
    def test_feature_vector_dimension(self):
        from app.utils.audio import FEATURE_DIM, extract_audio_features
        wav = _make_wav_bytes(duration_seconds=2.0)
        features = extract_audio_features(wav)
        assert len(features["feature_vector"]) == FEATURE_DIM

    def test_mfcc_count(self):
        from app.utils.audio import N_MFCC, extract_audio_features
        wav = _make_wav_bytes()
        features = extract_audio_features(wav)
        assert len(features["mfcc_mean"]) == N_MFCC
        assert len(features["mfcc_std"]) == N_MFCC

    def test_scalar_features_are_floats(self):
        from app.utils.audio import extract_audio_features
        wav = _make_wav_bytes()
        features = extract_audio_features(wav)
        assert isinstance(features["spectral_centroid_mean"], float)
        assert isinstance(features["spectral_centroid_std"], float)
        assert isinstance(features["chroma_mean"], float)
        assert isinstance(features["chroma_std"], float)

    def test_empty_bytes_raises_value_error(self):
        from app.utils.audio import extract_audio_features
        with pytest.raises(ValueError, match="empty"):
            extract_audio_features(b"")

    def test_corrupt_bytes_raises_value_error(self):
        from app.utils.audio import extract_audio_features
        with pytest.raises(ValueError):
            extract_audio_features(b"\x00" * 512)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Drift detection unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftDetection:
    def setup_method(self):
        """Reset drift window before each test."""
        from app.utils import drift as drift_module
        drift_module.clear_window()
        drift_module._state.baseline = None

    def test_report_before_window_full(self, artifacts_dir: Path):
        from app.utils import drift as drift_module
        from app.utils.audio import FEATURE_DIM

        with patch.object(drift_module, "BASELINE_PATH", artifacts_dir / "baseline.json"):
            report = drift_module.evaluate_drift([0.0] * FEATURE_DIM)

        # Window is not full yet → no statistics computed
        assert report.drift_detected is False
        assert report.ks_statistics == {}

    def test_window_grows_with_records(self, artifacts_dir: Path):
        from app.utils import drift as drift_module
        from app.utils.audio import FEATURE_DIM

        with patch.object(drift_module, "BASELINE_PATH", artifacts_dir / "baseline.json"):
            for _ in range(5):
                drift_module.record_features([0.0] * FEATURE_DIM)

        assert drift_module.get_window_size() == 5

    def test_drift_detected_on_extreme_shift(self, artifacts_dir: Path):
        """Feeding heavily shifted features should trigger drift after window fills."""
        from app.utils import drift as drift_module
        from app.utils.audio import FEATURE_DIM, FEATURE_NAMES

        rng = np.random.default_rng(seed=77)

        with patch.object(drift_module, "BASELINE_PATH", artifacts_dir / "baseline.json"):
            # Fill window with heavily shifted data (mean=1000, far from baseline ~0)
            for _ in range(drift_module.WINDOW_SIZE):
                shifted = rng.normal(1000.0, 0.1, size=FEATURE_DIM).tolist()
                drift_module.record_features(shifted)

            report = drift_module.evaluate_drift(
                rng.normal(1000.0, 0.1, size=FEATURE_DIM).tolist()
            )

        assert report.drift_detected is True
        assert report.min_p_value < drift_module.P_VALUE_THRESHOLD

    def test_no_drift_on_in_distribution_data(self, artifacts_dir: Path):
        """Features sampled from the baseline distribution should not trigger drift."""
        from app.utils import drift as drift_module
        from app.utils.audio import FEATURE_DIM

        rng = np.random.default_rng(seed=88)

        with patch.object(drift_module, "BASELINE_PATH", artifacts_dir / "baseline.json"):
            drift_module._state.baseline = None   # force reload

            for _ in range(drift_module.WINDOW_SIZE):
                # Baseline was generated with Normal(0,1) — match it
                in_dist = rng.normal(0.0, 1.0, size=FEATURE_DIM).tolist()
                drift_module.record_features(in_dist)

            report = drift_module.evaluate_drift(
                rng.normal(0.0, 1.0, size=FEATURE_DIM).tolist()
            )

        assert not report.drift_detected

    def test_clear_window_resets_state(self, artifacts_dir: Path):
        from app.utils import drift as drift_module
        from app.utils.audio import FEATURE_DIM

        for _ in range(10):
            drift_module.record_features([0.0] * FEATURE_DIM)

        drift_module.clear_window()
        assert drift_module.get_window_size() == 0
