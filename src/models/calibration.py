from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


@dataclass
class CalibratedEstimator:
    estimator: CalibratedClassifierCV
    method: str

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)


def fit_calibrator(base_estimator, x_val, y_val, method: str = "isotonic") -> CalibratedEstimator:
    calibrated = CalibratedClassifierCV(base_estimator, method=method, cv="prefit")
    calibrated.fit(x_val, y_val)
    return CalibratedEstimator(estimator=calibrated, method=method)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str]) -> float:
    idx = {c: i for i, c in enumerate(classes)}
    y_one_hot = np.zeros_like(y_prob)
    for i, label in enumerate(y_true.astype(str)):
        if label in idx:
            y_one_hot[i, idx[label]] = 1.0
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))


def binary_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))
