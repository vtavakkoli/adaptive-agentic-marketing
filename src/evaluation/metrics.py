from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(df: pd.DataFrame, preds: list[dict[str, Any]], latency_s: float) -> dict[str, Any]:
    y_true = (df["action_class"] == "do_nothing").astype(int)
    y_pred = pd.Series([1 if p["selected_action"] == "do_nothing" else 0 for p in preds])
    unnecessary = float(((y_pred == 0) & (df["no_action_preferred"] == 1)).mean())
    correct_no_action = float(((y_pred == 1) & (df["no_action_preferred"] == 1)).sum() / max((df["no_action_preferred"] == 1).sum(), 1))
    rule_viol = float(((y_pred == 0) & (df["fatigue_score"] > 0.66)).mean())
    fatigue_avoid = float(((y_pred == 1) & (df["fatigue_score"] > 0.66)).sum() / max((df["fatigue_score"] > 0.66).sum(), 1))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "unnecessary_contact_rate": unnecessary,
        "correct_no_action_rate": correct_no_action,
        "rule_violation_rate": rule_viol,
        "fatigue_avoidance_rate": fatigue_avoid,
        "latency_per_decision": float(latency_s / max(len(df), 1)),
    }


def timed_decisions(fn, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    preds = [fn(r) for r in rows]
    return preds, time.perf_counter() - start
