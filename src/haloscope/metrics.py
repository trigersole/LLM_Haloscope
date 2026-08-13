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

