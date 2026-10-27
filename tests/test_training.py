import numpy as np
import pytest

from src.evaluate import choose_f2_threshold, evaluate_predictions
from src.train import make_candidates


def test_candidate_pipelines_share_the_complete_preprocessing_path() -> None:
    candidates = make_candidates()
    assert {"dummy", "logistic_regression", "decision_tree", "random_forest", "xgboost"} <= set(candidates)
    assert all(
        [step for step, _ in pipeline.steps]
        == ["feature_engineering", "preprocessor", "model"]
        for pipeline in candidates.values()
    )


def test_threshold_selection_and_metrics_use_validation_cutoff() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.45, 0.7, 0.9])
    threshold, validation_f2 = choose_f2_threshold(y, probabilities)
    metrics = evaluate_predictions(y, probabilities, threshold)
    assert 0 <= threshold <= 1
    assert 0 <= validation_f2 <= 1
    assert metrics["false_negatives"] + metrics["true_positives"] == 3
    assert metrics["false_positives"] + metrics["true_negatives"] == 3
    assert 0 <= metrics["average_precision"] <= 1


def test_threshold_selection_rejects_one_class_validation_data() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        choose_f2_threshold([0, 0], [0.1, 0.2])
