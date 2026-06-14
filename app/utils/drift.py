"""
app/utils/drift.py
──────────────────
Statistical data-drift detection engine.

Strategy
--------
We maintain a rolling window of recent incoming feature vectors (one per
prediction request).  When the window reaches WINDOW_SIZE we run a
two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp) for every
feature dimension against the reference distribution stored in
``app/artifacts/baseline.json``.

A feature is flagged as "drifted" if its p-value falls below P_VALUE_THRESHOLD.
Overall drift is declared when the fraction of drifted features exceeds
DRIFT_FRACTION_THRESHOLD (Bonferroni-style relaxation).

The module exposes a thread-safe, module-level window so that concurrent
FastAPI workers share state without a database dependency.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
BASELINE_PATH = Path(__file__).parent.parent / "artifacts" / "baseline.json"

# Minimum number of incoming samples before a KS test is triggered.
WINDOW_SIZE: int = 30

# p-value below which a single feature is considered drifted.
P_VALUE_THRESHOLD: float = 0.05

# Fraction of features that must be drifted to raise an overall drift alert.
DRIFT_FRACTION_THRESHOLD: float = 0.25


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    """Result of a single drift evaluation run."""

    # Number of samples in the current evaluation window.
    window_size: int

    # Per-feature KS statistics keyed by feature name.
    ks_statistics: dict[str, float]

    # Per-feature p-values keyed by feature name.
    p_values: dict[str, float]

    # Features whose p-value is below the threshold.
    drifted_features: list[str]

    # Fraction of features that show drift.
    drift_fraction: float

    # True when drift_fraction ≥ DRIFT_FRACTION_THRESHOLD.
    drift_detected: bool

    # Minimum p-value across all features (worst-case signal).
    min_p_value: float

    @property
    def summary(self) -> str:
        status = "⚠ DRIFT DETECTED" if self.drift_detected else "✓ No drift"
        return (
            f"{status} | window={self.window_size} | "
            f"drifted={len(self.drifted_features)}/{len(self.p_values)} features | "
            f"min_p={self.min_p_value:.4f}"
        )


@dataclass
class _DriftState:
    """Module-level mutable state protected by a lock."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    window: list[list[float]] = field(default_factory=list)
    baseline: Optional[dict] = None


_state = _DriftState()


# ── Baseline loading ──────────────────────────────────────────────────────────

def _load_baseline() -> dict:
    """
    Load the baseline reference distribution from disk (cached after first load).

    Expected JSON schema::

        {
          "feature_names": ["mfcc_mean_0", ..., "chroma_std"],
          "reference_samples": {
            "mfcc_mean_0": [v1, v2, ...],
            ...
          }
        }
    """
    if _state.baseline is not None:
        return _state.baseline

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline file not found at {BASELINE_PATH}. "
            "Run `python scripts/generate_artifacts.py` to create it."
        )

    with BASELINE_PATH.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    required_keys = {"feature_names", "reference_samples"}
    if not required_keys.issubset(baseline.keys()):
        raise ValueError(
            f"baseline.json is missing required keys: {required_keys - set(baseline.keys())}"
        )

    _state.baseline = baseline
    logger.info(
        "Baseline loaded: %d features, %d reference samples each.",
        len(baseline["feature_names"]),
        len(next(iter(baseline["reference_samples"].values()))),
    )
    return baseline


def reload_baseline() -> None:
    """Force-reload the baseline from disk (useful after model retraining)."""
    with _state.lock:
        _state.baseline = None
    _load_baseline()


# ── Public API ────────────────────────────────────────────────────────────────

def record_features(feature_vector: list[float]) -> None:
    """
    Append a single prediction's feature vector to the rolling window.

    Parameters
    ----------
    feature_vector : list[float]
        The flat feature vector produced by ``extract_audio_features``.
    """
    with _state.lock:
        _state.window.append(feature_vector)
        # Evict old samples to bound memory usage (keep 2× window)
        if len(_state.window) > WINDOW_SIZE * 2:
            _state.window = _state.window[-WINDOW_SIZE:]


def evaluate_drift(feature_vector: list[float]) -> DriftReport:
    """
    Record the incoming feature vector and evaluate drift against baseline.

    If fewer than WINDOW_SIZE samples have been collected the report is
    returned with ``drift_detected=False`` and empty statistics, reflecting
    insufficient data rather than absence of drift.

    Parameters
    ----------
    feature_vector : list[float]
        Flat feature vector from the current prediction request.

    Returns
    -------
    DriftReport
        Full drift evaluation report.
    """
    record_features(feature_vector)
    baseline = _load_baseline()

    with _state.lock:
        current_window = list(_state.window)  # snapshot

    feature_names: list[str] = baseline["feature_names"]
    reference_samples: dict[str, list[float]] = baseline["reference_samples"]

    n_samples = len(current_window)

    if n_samples < WINDOW_SIZE:
        logger.debug(
            "Drift evaluation skipped: only %d/%d samples collected.",
            n_samples,
            WINDOW_SIZE,
        )
        return DriftReport(
            window_size=n_samples,
            ks_statistics={},
            p_values={},
            drifted_features=[],
            drift_fraction=0.0,
            drift_detected=False,
            min_p_value=1.0,
        )

    # Transpose window: shape (n_samples, n_features) → per-feature arrays
    incoming_array = np.array(current_window[-WINDOW_SIZE:])   # (W, F)

    ks_statistics: dict[str, float] = {}
    p_values: dict[str, float] = {}

    for feat_idx, feat_name in enumerate(feature_names):
        if feat_idx >= incoming_array.shape[1]:
            continue  # guard against dimension mismatch

        incoming_vals = incoming_array[:, feat_idx]
        reference_vals = np.array(reference_samples.get(feat_name, []))

        if len(reference_vals) < 2:
            logger.warning("Feature '%s' has no valid reference samples; skipping.", feat_name)
            continue

        ks_stat, p_val = stats.ks_2samp(incoming_vals, reference_vals)
        ks_statistics[feat_name] = float(ks_stat)
        p_values[feat_name] = float(p_val)

    drifted = [fn for fn, pv in p_values.items() if pv < P_VALUE_THRESHOLD]
    drift_fraction = len(drifted) / len(p_values) if p_values else 0.0
    drift_detected = drift_fraction >= DRIFT_FRACTION_THRESHOLD
    min_p = min(p_values.values()) if p_values else 1.0

    report = DriftReport(
        window_size=n_samples,
        ks_statistics=ks_statistics,
        p_values=p_values,
        drifted_features=drifted,
        drift_fraction=drift_fraction,
        drift_detected=drift_detected,
        min_p_value=min_p,
    )

    if drift_detected:
        logger.warning("DATA DRIFT DETECTED: %s", report.summary)
    else:
        logger.debug("Drift evaluation: %s", report.summary)

    return report


def get_window_size() -> int:
    """Return the current number of samples in the rolling window."""
    with _state.lock:
        return len(_state.window)


def clear_window() -> None:
    """Reset the rolling window (useful in tests)."""
    with _state.lock:
        _state.window.clear()
