import json

import joblib
import pandas as pd
import pytest

from src.predict import ModelBundle, load_model_bundle, predict_record
from src.preprocessing import FEATURE_COLUMNS, make_pipeline
from sklearn.dummy import DummyClassifier


def _fitted_pipeline():
    rows = pd.DataFrame(
        [
            ["L", 298.1, 308.6, 1551, 42.8, 0],
            ["M", 300.0, 310.0, 1400, 50.0, 30],
        ],
        columns=FEATURE_COLUMNS,
    )
    return make_pipeline(DummyClassifier(strategy="prior")).fit(rows, [0, 1])


def test_model_artifacts_load_and_predict(tmp_path) -> None:
    joblib.dump(_fitted_pipeline(), tmp_path / "predictive_maintenance.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "model_version": "test-version",
                "feature_columns": list(FEATURE_COLUMNS),
                "validation_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    bundle = load_model_bundle(tmp_path)
    record = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }
    prediction = predict_record(record, bundle)
    assert prediction["model_version"] == "test-version"
    assert prediction["risk_category"] in {"low", "high"}
    assert 0 <= prediction["failure_probability"] <= 1


def test_model_loader_requires_both_artifacts(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Run `python -m src.train`"):
        load_model_bundle(tmp_path)


def test_prediction_rejects_missing_and_unexpected_features() -> None:
    bundle = ModelBundle(
        pipeline=_fitted_pipeline(),
        metadata={"validation_threshold": 0.5, "model_version": "test"},
    )
    with pytest.raises(ValueError, match="Missing required features"):
        predict_record({"Type": "L"}, bundle)
    record = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
        "Machine failure": 1,
    }
    with pytest.raises(ValueError, match="Unexpected features"):
        predict_record(record, bundle)
