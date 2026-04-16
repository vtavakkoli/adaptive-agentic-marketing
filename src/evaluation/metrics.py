from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(df: pd.DataFrame, preds: list[dict[str, Any]], latency_s: float) -> dict[str, Any]:
    pred_actions = pd.Series([p["selected_action"] for p in preds], name="predicted_action")
    y_true_no_action = (df["action_class"] == "do_nothing").astype(int)
    y_pred_no_action = (pred_actions == "do_nothing").astype(int)

    unnecessary = float(((y_pred_no_action == 0) & (df["no_action_preferred"] == 1)).mean())
    correct_no_action = float(
        ((y_pred_no_action == 1) & (df["no_action_preferred"] == 1)).sum() / max((df["no_action_preferred"] == 1).sum(), 1)
    )
    rule_viol = float(((y_pred_no_action == 0) & (df["fatigue_score"] > 0.66)).mean())
    fatigue_avoid = float(
        ((y_pred_no_action == 1) & (df["fatigue_score"] > 0.66)).sum() / max((df["fatigue_score"] > 0.66).sum(), 1)
    )

    labels = sorted(set(df["action_class"].astype(str).unique()) | set(pred_actions.astype(str).unique()))
    multi_cm = confusion_matrix(df["action_class"], pred_actions, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true_no_action, y_pred_no_action)),
        "precision": float(precision_score(y_true_no_action, y_pred_no_action, zero_division=0)),
        "recall": float(recall_score(y_true_no_action, y_pred_no_action, zero_division=0)),
        "f1": float(f1_score(y_true_no_action, y_pred_no_action, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_no_action, y_pred_no_action).tolist(),
        "unnecessary_contact_rate": unnecessary,
        "correct_no_action_rate": correct_no_action,
        "rule_violation_rate": rule_viol,
        "fatigue_avoidance_rate": fatigue_avoid,
        "latency_per_decision": float(latency_s / max(len(df), 1)),
        "action_class_distribution_true": df["action_class"].value_counts().to_dict(),
        "action_class_distribution_pred": pred_actions.value_counts().to_dict(),
        "no_action_distribution": {
            "true_no_action_rate": float((df["action_class"] == "do_nothing").mean()),
            "pred_no_action_rate": float((pred_actions == "do_nothing").mean()),
        },
        "multiclass": {
            "labels": labels,
            "accuracy": float(accuracy_score(df["action_class"], pred_actions)),
            "macro_f1": float(f1_score(df["action_class"], pred_actions, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(df["action_class"], pred_actions, labels=labels, average="weighted", zero_division=0)
            ),
            "confusion_matrix": multi_cm.tolist(),
            "per_action_true_counts": df["action_class"].value_counts().to_dict(),
            "per_action_pred_counts": pred_actions.value_counts().to_dict(),
        },
    }


def timed_decisions(fn, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    preds = [fn(r) for r in rows]
    return preds, time.perf_counter() - start
