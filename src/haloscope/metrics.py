"""Small NumPy-only binary metrics to keep the laptop path dependency-light."""

from __future__ import annotations

import numpy as np


def _inputs(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("metrics require matching arrays containing both binary classes")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")
    return labels, scores


def roc_auc(labels, scores) -> float:
    """Mann–Whitney AUROC with average ranks for tied scores."""
    labels, scores = _inputs(labels, scores)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = len(labels) - n_positive
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def average_precision(labels, scores) -> float:
    labels, scores = _inputs(labels, scores)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_true = np.cumsum(sorted_labels)
    precision = cumulative_true / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / sorted_labels.sum())


def binary_accuracy(labels, predictions) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions have different lengths")
    return float(np.mean(labels == predictions))


def balanced_accuracy(labels, predictions) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if len(labels) != len(predictions) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("balanced accuracy requires matching binary arrays")
    true_positive_rate = np.mean(predictions[labels == 1] == 1)
    true_negative_rate = np.mean(predictions[labels == 0] == 0)
    return float((true_positive_rate + true_negative_rate) / 2.0)


def brier_score(labels, probabilities) -> float:
    labels, probabilities = _inputs(labels, probabilities)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return float(np.mean((probabilities - labels) ** 2))


def expected_calibration_error(labels, probabilities, bins: int = 10) -> float:
    labels, probabilities = _inputs(labels, probabilities)
    if bins < 1 or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("bins must be positive and probabilities must be in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if np.any(selected):
            accuracy = labels[selected].mean()
            confidence = probabilities[selected].mean()
            total += selected.mean() * abs(float(accuracy - confidence))
    return float(total)


def fpr_at_tpr(labels, scores, target_tpr: float = 0.95) -> float:
    labels, scores = _inputs(labels, scores)
    if not 0.0 < target_tpr <= 1.0:
        raise ValueError("target_tpr must be in (0, 1]")
    positive_scores = np.sort(scores[labels == 1])[::-1]
    required = max(1, int(np.ceil(target_tpr * len(positive_scores))))
    threshold = positive_scores[required - 1]
    return float(np.mean(scores[labels == 0] >= threshold))

