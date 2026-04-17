from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.metrics import evaluate_predictions


def test_metrics_include_per_class_and_confusion() -> None:
    df = pd.DataFrame(
        {
            "action_class": ["do_nothing", "send_information", "send_reminder", "defer_action"],
            "no_action_preferred": [1, 0, 0, 0],
            "fatigue_score": [0.8, 0.2, 0.2, 0.1],
        }
    )
    preds = [
        {"selected_action": "do_nothing", "no_action": True},
        {"selected_action": "send_information", "no_action": False},
        {"selected_action": "send_reminder", "no_action": False},
        {"selected_action": "defer_action", "no_action": False},
    ]

    metrics = evaluate_predictions(df, preds, latency_s=0.2)
    assert "balanced_accuracy" in metrics["multiclass"]
    assert "per_class" in metrics["multiclass"]
    assert "send_information" in metrics["multiclass"]["per_class"]
    assert "prediction_diversity" in metrics


def test_metrics_raise_for_no_action_mismatch() -> None:
    df = pd.DataFrame(
        {
            "action_class": ["send_information"],
            "no_action_preferred": [0],
            "fatigue_score": [0.1],
        }
    )
    preds = [{"selected_action": "send_information", "no_action": True}]
    with pytest.raises(ValueError):
        evaluate_predictions(df, preds, latency_s=0.01)
