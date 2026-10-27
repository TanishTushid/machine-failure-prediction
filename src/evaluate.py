"""Evaluation metrics and validation-only decision threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_f2_threshold(y_true: Any, probabilities: Any) -> tuple[float, float]:
    """Choose the validation threshold that maximizes F2.

    F2 weights recall more heavily than precision, reflecting the stated
    preference to avoid missed failures. In a tie, the higher threshold is
    chosen to reduce false alarms. This function must not receive test labels.
    """
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and probabilities must be equal-length 1D arrays.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if set(np.unique(labels)) - {0, 1} or len(np.unique(labels)) < 2:
        raise ValueError("Threshold selection requires both binary classes.")

    thresholds = np.unique(np.concatenate(([0.0], scores, [1.0])))
    candidate_scores = [
        fbeta_score(labels, scores >= threshold, beta=2, zero_division=0)
        for threshold in thresholds
    ]
    best_f2 = max(candidate_scores)
    best_threshold = max(
        threshold
        for threshold, score in zip(thresholds, candidate_scores)
        if np.isclose(score, best_f2)
    )
    return float(best_threshold), float(best_f2)


def evaluate_predictions(
    y_true: Any, probabilities: Any, threshold: float
) -> dict[str, float | int]:
    """Calculate failure-sensitive metrics and confusion-matrix counts."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and probabilities must be equal-length 1D arrays.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }
