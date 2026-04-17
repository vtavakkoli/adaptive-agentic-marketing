import numpy as np

from src.models.calibration import expected_calibration_error, multiclass_brier_score


def test_calibration_metrics_shapes() -> None:
    y_true = np.array(["a", "b", "a"])
    y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.55, 0.45]])
    brier = multiclass_brier_score(y_true, y_prob, ["a", "b"])
    assert brier >= 0.0

    ece = expected_calibration_error(np.array([1, 1, 0]), np.array([0.9, 0.6, 0.55]))
    assert 0.0 <= ece <= 1.0
