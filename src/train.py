"""Train, compare, select, and persist the predictive-maintenance model."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_loader import EXPECTED_MACHINE_TYPES, load_dataset
from src.evaluate import choose_f2_threshold, evaluate_predictions
from src.preprocessing import FEATURE_COLUMNS, TARGET, make_pipeline

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42
MODEL_VERSION = "1.0.0"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def make_candidates(scale_pos_weight: float = 28.5) -> dict[str, Any]:
    """Return comparable baseline classifiers, each inside the same pipeline."""
    return {
        "dummy": make_pipeline(
            DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
        ),
        "logistic_regression": make_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1500,
                random_state=RANDOM_STATE,
            )
        ),
        "decision_tree": make_pipeline(
            DecisionTreeClassifier(
                class_weight="balanced",
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            )
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=200,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "xgboost": make_pipeline(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                tree_method="hist",
                n_jobs=4,
                random_state=RANDOM_STATE,
            )
        ),
    }


def _tune_candidates(
    candidates: dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, cv: Any
) -> dict[str, tuple[Any, float]]:
    """Tune compact randomized spaces for the two tree-ensemble baselines."""
    spaces = {
        "random_forest": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [None, 8, 14],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.8],
        },
        "xgboost": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [3, 4, 6],
            "model__learning_rate": [0.03, 0.07, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
    }
    tuned: dict[str, tuple[Any, float]] = {}
    for name, parameter_space in spaces.items():
        search = RandomizedSearchCV(
            estimator=candidates[name],
            param_distributions=parameter_space,
            n_iter=4,
            scoring="average_precision",
            cv=cv,
            refit=False,
            random_state=RANDOM_STATE,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_pipeline = clone(candidates[name]).set_params(**search.best_params_)
        tuned[f"{name}_tuned"] = (best_pipeline, float(search.best_score_))
        LOGGER.info(
            "%s tuned CV average precision %.4f; params=%s",
            name,
            search.best_score_,
            search.best_params_,
        )
    return tuned


def _save_evaluation_artifacts(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    metrics: dict[str, float | int],
    artifact_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

    artifact_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        probabilities >= threshold,
        labels=[0, 1],
        display_labels=["No failure", "Failure"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Test confusion matrix (threshold={threshold:.3f})")
    fig.tight_layout()
    fig.savefig(artifact_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.set_title("Test precision-recall curve")
    fig.tight_layout()
    fig.savefig(artifact_dir / "precision_recall_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame([metrics]).to_csv(artifact_dir / "test_metrics.csv", index=False)


def _log_mlflow(
    model_name: str,
    cv_score: float,
    threshold: float,
    metrics: dict[str, float | int],
    model: Any,
) -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        warnings.warn(
            "MLflow is not installed; experiment tracking was skipped. "
            "Install project dependencies to enable it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    estimator_parameters = {
        f"estimator.{key}": str(value)
        for key, value in model.named_steps["model"].get_params(deep=False).items()
    }
    mlflow.set_experiment("predictive-maintenance-failure")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(
            {
                "selected_model": model_name,
                "random_state": RANDOM_STATE,
                "cv_scoring": "average_precision",
                "threshold_criterion": "maximize validation F2; tie -> higher threshold",
                "threshold": threshold,
                "model_version": MODEL_VERSION,
                **estimator_parameters,
            }
        )
        mlflow.log_metric("cv_average_precision", cv_score)
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.sklearn.log_model(model, artifact_path="model")


def train_and_evaluate(
    data_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    tune: bool = True,
    track: bool = True,
) -> dict[str, Any]:
    """Fit on train only, select using CV/validation, and evaluate test once."""
    dataframe = load_dataset(data_path)
    if dataframe[TARGET].isna().any():
        raise ValueError(f"Target column {TARGET!r} contains missing values.")
    X = dataframe.loc[:, FEATURE_COLUMNS].copy()
    y = dataframe[TARGET].astype(int).copy()

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    if positive_count == 0:
        raise ValueError("Training partition contains no failure examples.")
    candidates = make_candidates(negative_count / positive_count)

    scores: dict[str, tuple[Any, float]] = {}
    for name, pipeline in candidates.items():
        fold_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="average_precision",
            n_jobs=1,
            error_score="raise",
        )
        scores[name] = (pipeline, float(np.mean(fold_scores)))
        LOGGER.info(
            "%s CV average precision %.4f (+/- %.4f)",
            name,
            np.mean(fold_scores),
            np.std(fold_scores),
        )

    if tune:
        scores.update(_tune_candidates(candidates, X_train, y_train, cv))

    leaderboard = pd.DataFrame(
        [
            {"model": name, "cv_average_precision": score}
            for name, (_, score) in scores.items()
        ]
    ).sort_values("cv_average_precision", ascending=False, ignore_index=True)
    best_name = str(leaderboard.iloc[0]["model"])
    best_pipeline, best_cv_score = scores[best_name]
    best_pipeline.fit(X_train, y_train)

    validation_probabilities = best_pipeline.predict_proba(X_validation)[:, 1]
    threshold, validation_f2 = choose_f2_threshold(
        y_validation, validation_probabilities
    )
    test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probabilities, threshold)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "predictive_maintenance.joblib"
    metadata_path = output_path / "metadata.json"
    joblib.dump(best_pipeline, model_path)
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_note": "Synthetic AI4I 2020 dataset; not evidence of real-world performance.",
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_schema": {
            "Type": {
                "type": "category",
                "allowed_values": sorted(EXPECTED_MACHINE_TYPES),
            },
            **{
                feature: {"type": "number"}
                for feature in FEATURE_COLUMNS
                if feature != "Type"
            },
        },
        "target": TARGET,
        "selected_model": best_name,
        "selected_estimator_parameters": {
            key: str(value)
            for key, value in best_pipeline.named_steps["model"]
            .get_params(deep=False)
            .items()
        },
        "selection_metric": "3-fold stratified training average precision",
        "cross_validation_average_precision": best_cv_score,
        "validation_threshold": threshold,
        "threshold_criterion": "Maximize validation F2 (recall-weighted); ties choose higher threshold.",
        "validation_f2": validation_f2,
        "test_metrics": test_metrics,
        "test_set_role": "Single final evaluation after model and threshold selection.",
        "probability_note": "Uncalibrated model score; not a calibrated real-world failure probability.",
        "artifact_format": "joblib; load only trusted artifacts.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(Path(artifact_dir) / "model_comparison.csv", index=False)
    _save_evaluation_artifacts(
        y_test, test_probabilities, threshold, test_metrics, Path(artifact_dir)
    )
    if track:
        _log_mlflow(best_name, best_cv_score, threshold, test_metrics, best_pipeline)

    LOGGER.info("Selected %s with CV AP %.4f", best_name, best_cv_score)
    LOGGER.info("Validation F2 %.4f at threshold %.4f", validation_f2, threshold)
    LOGGER.info("Final test metrics: %s", test_metrics)
    return {
        "leaderboard": leaderboard,
        "metadata": metadata,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="Path to ai4i2020.csv")
    parser.add_argument("--no-tune", action="store_true", help="Skip randomized tuning")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = train_and_evaluate(
        data_path=args.data,
        tune=not args.no_tune,
        track=not args.no_mlflow,
    )
    print("\nModel comparison (training-only stratified CV):")
    print(result["leaderboard"].to_string(index=False))
    print("\nSelected model:", result["metadata"]["selected_model"])
    print("Validation-selected threshold:", result["metadata"]["validation_threshold"])
    print("Final test metrics:")
    print(json.dumps(result["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
