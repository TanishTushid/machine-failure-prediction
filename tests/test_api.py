import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from app.main import app, require_model_bundle
from src.predict import ModelBundle
from src.preprocessing import FEATURE_COLUMNS, make_pipeline


def _fake_bundle() -> ModelBundle:
    samples = [
        ["L", 298.1, 308.6, 1551, 42.8, 0],
        ["M", 300.0, 310.0, 1400, 50.0, 30],
    ]
    pipeline = make_pipeline(DummyClassifier(strategy="prior"))
    pipeline.fit(
        pd.DataFrame(samples, columns=FEATURE_COLUMNS),
        [0, 1],
    )
    return ModelBundle(
        pipeline=pipeline,
        metadata={
            "model_version": "test",
            "selected_model": "dummy",
            "validation_threshold": 0.5,
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_schema": {},
            "probability_note": "test score",
            "dataset_note": "synthetic test",
        },
    )


def test_health_and_prediction_response() -> None:
    app.dependency_overrides[require_model_bundle] = _fake_bundle
    try:
        client = TestClient(app)
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post(
            "/predict",
            json={
                "Type": "M",
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 0,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in (0, 1)
        assert 0 <= body["failure_probability"] <= 1
        assert body["risk_category"] in ("low", "high")
        assert body["model_version"] == "test"
        assert client.get("/model-info").json()["model_version"] == "test"
    finally:
        app.dependency_overrides.clear()


def test_invalid_api_payloads_are_rejected() -> None:
    app.dependency_overrides[require_model_bundle] = _fake_bundle
    client = TestClient(app)
    base = {
        "Type": "M",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 0,
    }
    try:
        missing = dict(base)
        missing.pop("torque_nm")
        assert client.post("/predict", json=missing).status_code == 422
        invalid = {**base, "air_temperature_k": -1}
        assert client.post("/predict", json=invalid).status_code == 422
        non_finite = {**base, "air_temperature_k": "NaN"}
        assert client.post("/predict", json=non_finite).status_code == 422
        unexpected = {**base, "Type": "X"}
        assert client.post("/predict", json=unexpected).status_code == 422
        extra = {**base, "TWF": 1}
        assert client.post("/predict", json=extra).status_code == 422
    finally:
        app.dependency_overrides.clear()
