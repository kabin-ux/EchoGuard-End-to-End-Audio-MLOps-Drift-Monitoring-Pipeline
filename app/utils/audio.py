"""
app/utils/audio.py
──────────────────
Librosa-based feature extraction pipeline.

Feature vector layout (30 features total):
  [0:13]  – MFCC means          (13 coefficients)
  [13:26] – MFCC stds           (13 coefficients)
  [26]    – Spectral Centroid mean
  [27]    – Spectral Centroid std
  [28]    – Chroma STFT mean    (averaged across 12 pitch bins → scalar)
  [29]    – Chroma STFT std     (averaged across 12 pitch bins → scalar)

We deliberately keep the vector compact so that the KS drift test
has enough statistical power on small batches.
"""

from __future__ import annotations

import io
import logging
from typing import TypedDict

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
N_MFCC = 13
SAMPLE_RATE = 22_050      # librosa default; resample everything to this rate
FEATURE_NAMES: list[str] = (
    [f"mfcc_mean_{i}" for i in range(N_MFCC)]
    + [f"mfcc_std_{i}" for i in range(N_MFCC)]
    + ["spectral_centroid_mean", "spectral_centroid_std"]
    + ["chroma_mean", "chroma_std"]
)
FEATURE_DIM = len(FEATURE_NAMES)   # 30


class AudioFeatures(TypedDict):
    """Structured representation of extracted audio features."""

    mfcc_mean: list[float]          # shape (13,)
    mfcc_std: list[float]           # shape (13,)
    spectral_centroid_mean: float
    spectral_centroid_std: float
    chroma_mean: float              # single scalar (mean over 12 chroma bins)
    chroma_std: float               # single scalar (std over 12 chroma bins)
    feature_vector: list[float]     # flat 30-element list for the model


# ── Public API ────────────────────────────────────────────────────────────────

def extract_audio_features(file_bytes: bytes) -> AudioFeatures:
    """
    Load audio from raw bytes and compute a fixed-length feature vector.

    Parameters
    ----------
    file_bytes : bytes
        Raw audio file content (WAV, MP3, FLAC, OGG …).

    Returns
    -------
    AudioFeatures
        Typed dict containing per-feature statistics and the flat feature vector.

    Raises
    ------
    ValueError
        If the audio cannot be decoded or produces an empty signal.
    RuntimeError
        If feature extraction fails for any unexpected reason.
    """
    if not file_bytes:
        raise ValueError("Received empty audio payload.")

    logger.debug("Loading audio from memory buffer (%d bytes).", len(file_bytes))

    try:
        audio_buf = io.BytesIO(file_bytes)
        y, sr = librosa.load(audio_buf, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise ValueError(f"Failed to decode audio: {exc}") from exc

    if len(y) == 0:
        raise ValueError("Audio signal is empty after decoding.")

    logger.debug("Audio loaded: %.2f s at %d Hz.", len(y) / sr, sr)

    try:
        features = _compute_features(y, sr)
    except Exception as exc:
        raise RuntimeError(f"Feature extraction failed: {exc}") from exc

    return features


# ── Private helpers ───────────────────────────────────────────────────────────

def _compute_features(y: np.ndarray, sr: int) -> AudioFeatures:
    """Compute all acoustic features from a mono waveform."""

    # ── MFCCs ────────────────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)   # (13, T)
    mfcc_mean = mfccs.mean(axis=1)                              # (13,)
    mfcc_std = mfccs.std(axis=1)                                # (13,)

    # ── Spectral Centroid ─────────────────────────────────────────────────────
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # (1, T)
    sc_mean = float(spec_centroid.mean())
    sc_std = float(spec_centroid.std())

    # ── Chroma STFT ───────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)            # (12, T)
    # Collapse 12 pitch classes → single mean / std to keep vector compact
    chroma_mean = float(chroma.mean())
    chroma_std = float(chroma.std())

    # ── Assemble flat feature vector ──────────────────────────────────────────
    feature_vector: np.ndarray = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            [sc_mean, sc_std],
            [chroma_mean, chroma_std],
        ]
    )

    assert feature_vector.shape == (FEATURE_DIM,), (
        f"Feature vector size mismatch: expected {FEATURE_DIM}, "
        f"got {feature_vector.shape[0]}"
    )

    return AudioFeatures(
        mfcc_mean=mfcc_mean.tolist(),
        mfcc_std=mfcc_std.tolist(),
        spectral_centroid_mean=sc_mean,
        spectral_centroid_std=sc_std,
        chroma_mean=chroma_mean,
        chroma_std=chroma_std,
        feature_vector=feature_vector.tolist(),
    )


def feature_names() -> list[str]:
    """Return the ordered list of feature names matching the feature vector."""
    return FEATURE_NAMES.copy()
