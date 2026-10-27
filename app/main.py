"""FastAPI service for predictive-maintenance inference."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import ModelBundle, get_model_bundle, predict_record

app = FastAPI(
    title="Predictive Maintenance Failure API",
    description=(
        "Failure-risk estimates from a model trained on synthetic AI4I data. "
        "Scores are not calibrated real-world failure probabilities."
    ),
    version="1.0.0",
)


class SensorInput(BaseModel):
    """Validated request schema containing only leakage-safe sensor inputs."""

    model_config = ConfigDict(extra="forbid")

    Type: Literal["L", "M", "H"]
    air_temperature_k: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    process_temperature_k: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    rotational_speed_rpm: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    torque_nm: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    tool_wear_min: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    def to_features(self) -> dict[str, str | float]:
        return {
            "Type": self.Type,
            "Air temperature [K]": self.air_temperature_k,
            "Process temperature [K]": self.process_temperature_k,
            "Rotational speed [rpm]": self.rotational_speed_rpm,
            "Torque [Nm]": self.torque_nm,
            "Tool wear [min]": self.tool_wear_min,
        }


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    failure_probability: Annotated[float, Field(ge=0, le=1)]
    risk_category: Literal["low", "high"]
    model_version: str


def require_model_bundle() -> ModelBundle:
    """Load model artifacts, translating absent artifacts to an API error."""
    try:
        return get_model_bundle()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/health")
def health() -> dict[str, str]:
    """Return service liveness without forcing model artifact loading."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info(bundle: ModelBundle = Depends(require_model_bundle)) -> dict[str, object]:
    """Expose selected model version, threshold, and schema."""
    return {
        "model_version": bundle.metadata["model_version"],
        "selected_model": bundle.metadata["selected_model"],
        "threshold": bundle.metadata["validation_threshold"],
        "feature_columns": bundle.metadata["feature_columns"],
        "feature_schema": bundle.metadata["feature_schema"],
        "probability_note": bundle.metadata["probability_note"],
        "dataset_note": bundle.metadata["dataset_note"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: SensorInput,
    bundle: ModelBundle = Depends(require_model_bundle),
) -> PredictionResponse:
    """Predict failure class and threshold-based risk from sensor readings."""
    result = predict_record(request.to_features(), bundle)
    return PredictionResponse(**result)
