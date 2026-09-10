import numpy as np

from haloscope.metrics import (
    balanced_accuracy,
    brier_score,
    expected_calibration_error,
    fpr_at_tpr,
)


def test_extended_metrics_perfect_predictions():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.1, 0.9, 1.0])
    predictions = probabilities >= 0.5
    assert balanced_accuracy(labels, predictions) == 1.0
    assert np.isclose(brier_score(labels, probabilities), 0.005)
    assert np.isclose(expected_calibration_error(labels, probabilities), 0.05)
    assert fpr_at_tpr(labels, probabilities) == 0.0
