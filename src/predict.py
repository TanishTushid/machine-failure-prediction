"""Trusted-artifact loading and validated single-record inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class ModelBundle:
    """A fitted pipeline paired with its decision metadata."""

    pipeline: Any
    metadata: dict[str, Any]


def load_model_bundle(model_dir: str | Path = MODEL_DIR) -> ModelBundle:
    """Load a joblib pipeline and metadata from a trusted local directory.

    Joblib uses Python pickle internally. Never load a model artifact from an
    untrusted source.
    """
    directory = Path(model_dir)
    model_path = directory / "predictive_maintenance.joblib"
    metadata_path = directory / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model artifacts are missing from {directory}. Run "
            "`python -m src.train` before starting the API."
        )

    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Model metadata feature schema does not match this code.")
    if "validation_threshold" not in metadata or "model_version" not in metadata:
        raise ValueError("Model metadata is missing threshold or version.")
    return ModelBundle(pipeline=pipeline, metadata=metadata)


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    """Load the configured model once per application process."""
    return load_model_bundle(MODEL_DIR)


def predict_record(features: Mapping[str, Any], bundle: ModelBundle) -> dict[str, Any]:
    """Return class, uncalibrated failure score, and threshold-based risk."""
    missing = sorted(set(FEATURE_COLUMNS) - set(features))
    unexpected = sorted(set(features) - set(FEATURE_COLUMNS))
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected features: {', '.join(unexpected)}")

    row = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
    probability = float(bundle.pipeline.predict_proba(row)[0, 1])
    threshold = float(bundle.metadata["validation_threshold"])
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "failure_probability": probability,
        "risk_category": "high" if prediction else "low",
        "model_version": str(bundle.metadata["model_version"]),
    }
