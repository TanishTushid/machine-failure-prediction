import pandas as pd
import pytest

from src.preprocessing import FEATURE_COLUMNS, PhysicalFeatureEngineer, make_pipeline
from sklearn.linear_model import LogisticRegression


def example_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["M", 298.1, 308.6, 1551, 42.8, 0],
            ["L", 300.0, 310.0, 1400, 50.0, 30],
        ],
        columns=FEATURE_COLUMNS,
    )


def test_engineering_adds_hypothesis_driven_features() -> None:
    transformed = PhysicalFeatureEngineer().fit_transform(example_features())
    assert transformed.loc[0, "Temperature difference [K]"] == pytest.approx(10.5)
    assert transformed.loc[0, "Mechanical power proxy [rpm·Nm]"] == pytest.approx(
        1551 * 42.8
    )


def test_engineer_rejects_missing_feature() -> None:
    with pytest.raises(ValueError, match="Missing required feature"):
        PhysicalFeatureEngineer().fit_transform(example_features().drop(columns="Type"))


def test_pipeline_ignores_extra_columns_and_predicts_probabilities() -> None:
    X = example_features()
    X["TWF"] = [0, 0]
    y = [0, 1]
    pipeline = make_pipeline(LogisticRegression())
    pipeline.fit(X.loc[:, FEATURE_COLUMNS], y)
    probabilities = pipeline.predict_proba(X.loc[:, FEATURE_COLUMNS])
    assert probabilities.shape == (2, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
