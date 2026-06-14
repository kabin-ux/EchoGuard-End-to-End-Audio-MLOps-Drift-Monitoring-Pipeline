"""
scripts/generate_artifacts.py
──────────────────────────────
Generates two artifacts required by the FastAPI application:

  1. app/artifacts/model.joblib
     A scikit-learn RandomForestClassifier trained on REAL GTZAN audio
     feature distributions (30-dimensional feature vectors).

  2. app/artifacts/baseline.json
     Reference feature distributions drawn from a hold-out validation slice
     of the dataset, used by the KS drift detector at inference time.

Usage
─────
  python scripts/generate_artifacts.py /path/to/genres_original
"""

from __future__ import annotations

import json
import logging
import sys
import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "app" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
BASELINE_PATH = ARTIFACTS_DIR / "baseline.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GENRES: list[str] = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock",
]

FEATURE_DIM = 30  # 13 mean + 13 std MFCCs, 2 Centroid, 2 Chroma


def extract_audio_features(file_path: Path) -> np.ndarray | None:
    """
    Extracts 30 canonical audio features matching production inputs.
    Handles broken files safely (e.g., jazz.00054.wav in the original dataset).
    """
    try:
        # Load exactly 30 seconds sampled at 22050Hz matching standard GTZAN benchmarks
        y, sr = librosa.load(file_path, sr=22050, duration=30.0)
        if len(y) == 0:
            return None

        # 1. MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # 2. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)

        # 3. Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma)
        chroma_std = np.std(chroma)

        # Stitch into a flat 30-dimensional feature vector
        vector = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [centroid_mean, centroid_std],
            [chroma_mean, chroma_std]
        ])
        return vector.astype(np.float32)

    except Exception as e:
        logger.warning("Skipping corrupted file %s: %s", file_path.name, str(e))
        return None


def process_real_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Scans subdirectories for WAV audio targets and runs feature compilation."""
    X_parts = []
    y_parts = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path root not found at: {dataset_dir}")

    logger.info("Scanning dataset categories in %s...", dataset_dir)
    for genre in GENRES:
        genre_folder = dataset_dir / genre
        if not genre_folder.exists():
            logger.warning("Genre directory missing: %s", genre_folder)
            continue

        audio_files = list(genre_folder.glob("*.wav"))
        logger.info("Processing '%s' (%d audio tracks found)...", genre, len(audio_files))

        count = 0
        for track in audio_files:
            features = extract_audio_features(track)
            if features is not None:
                X_parts.append(features)
                y_parts.append(genre)
                count += 1
        
        logger.info("Successfully extracted %d tracks for '%s'", count, genre)

    if not X_parts:
        raise ValueError("No features were extracted. Verify your audio directory paths.")

    return np.vstack(X_parts), np.array(y_parts)


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[RandomForestClassifier, LabelEncoder, float, np.ndarray, np.ndarray]:
    """Train a RandomForest and return model assets alongside holdout validation sets."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 80% Train, 20% Test split strategy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=250,  # Bumped estimators slightly for more robust real feature boundary cuts
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    
    return clf, le, accuracy, X_test, y_test


def _build_feature_names() -> list[str]:
    """Generates the canonical label definitions mapping array indexing."""
    n_mfcc = 13
    return (
        [f"mfcc_mean_{i}" for i in range(n_mfcc)]
        + [f"mfcc_std_{i}" for i in range(n_mfcc)]
        + ["spectral_centroid_mean", "spectral_centroid_std"]
        + ["chroma_mean", "chroma_std"]
    )


def build_baseline(X_test: np.ndarray) -> dict:
    """
    Builds an authentic reference distribution for drift monitoring
    using a holdout test partition.
    """
    feature_names = _build_feature_names()
    reference_samples: dict[str, list[float]] = {}

    for feat_idx, feat_name in enumerate(feature_names):
        reference_samples[feat_name] = X_test[:, feat_idx].tolist()

    return {
        "feature_names": feature_names,
        "reference_samples": reference_samples,
        "metadata": {
            "n_baseline_samples": X_test.shape[0],
            "feature_dim": FEATURE_DIM,
            "genres": GENRES,
            "description": (
                "Real reference distribution mapping validation-split audio matrices "
                "for Kolmogorov-Smirnov inference drift scoring variables."
            ),
        },
    }


def save_model(clf: RandomForestClassifier, le: LabelEncoder) -> None:
    bundle = {"model": clf, "label_encoder": le, "genres": GENRES}
    joblib.dump(bundle, MODEL_PATH, compress=3)
    logger.info("Model saved → %s (%.1f KB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1024)


def save_baseline(baseline: dict) -> None:
    with BASELINE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2)
    logger.info("Baseline saved → %s (%.1f KB)", BASELINE_PATH, BASELINE_PATH.stat().st_size / 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features and compile training pipeline artifacts.")
    parser.add_argument("data_dir", type=str, help="Path to the root folder holding genre classes (e.g. genres_original/)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Production Artifact Processing Initiated")
    logger.info("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Process Real Data ──────────────────────────────────────────────────
    X, y = process_real_dataset(Path(args.data_dir))
    logger.info("Dataset feature grid extraction complete: X=%s, y=%s", X.shape, y.shape)

    # ── 2. Train Model ────────────────────────────────────────────────────────
    logger.info("Fitting RandomForestClassifier on real audio inputs...")
    clf, le, accuracy, X_test, _ = train_model(X, y)
    logger.info("Real Evaluation Test Accuracy: %.2f%%", accuracy * 100)
    save_model(clf, le)

    # ── 3. Establish Baseline from Holdouts ───────────────────────────────────
    logger.info("Compiling drift monitoring baseline matrices...")
    baseline = build_baseline(X_test)
    save_baseline(baseline)

    logger.info("=" * 60)
    logger.info("Done! Active assets compiled ready inside %s", ARTIFACTS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()