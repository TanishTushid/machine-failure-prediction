"""Leakage-safe feature engineering and preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "Machine failure"
FEATURE_COLUMNS = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
NUMERIC_FEATURES = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature difference [K]",
    "Mechanical power proxy [rpm·Nm]",
)
CATEGORICAL_FEATURES = ("Type",)


class PhysicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add temperature difference and a speed-times-torque proxy."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> PhysicalFeatureEngineer:
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X)
        engineered = X.loc[:, FEATURE_COLUMNS].copy()
        engineered["Temperature difference [K]"] = (
            engineered["Process temperature [K]"] - engineered["Air temperature [K]"]
        )
        engineered["Mechanical power proxy [rpm·Nm]"] = (
            engineered["Rotational speed [rpm]"] * engineered["Torque [Nm]"]
        )
        return engineered

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return [*FEATURE_COLUMNS, "Temperature difference [K]", "Mechanical power proxy [rpm·Nm]"]

    @staticmethod
    def _validate_columns(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input features must be a pandas DataFrame.")
        missing = sorted(set(FEATURE_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required feature columns: {', '.join(missing)}")


def make_preprocessor() -> ColumnTransformer:
    """Create preprocessing fitted only on the training partition."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def make_pipeline(classifier: Any) -> Pipeline:
    """Build the complete engineering, preprocessing, and estimator pipeline."""
    return Pipeline(
        steps=[
            ("feature_engineering", PhysicalFeatureEngineer()),
            ("preprocessor", make_preprocessor()),
            ("model", classifier),
        ]
    )
