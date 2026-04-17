from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def _entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.array([v / total for v in counts.values() if v > 0], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def _l1_distribution_shift(true_counts: dict[str, int], pred_counts: dict[str, int], labels: list[str]) -> float:
    true_total = sum(true_counts.values())
    pred_total = sum(pred_counts.values())
    if true_total == 0 or pred_total == 0:
        return 0.0
    shift = 0.0
    for label in labels:
        p_true = true_counts.get(label, 0) / true_total
        p_pred = pred_counts.get(label, 0) / pred_total
        shift += abs(p_true - p_pred)
    return float(0.5 * shift)


def evaluate_predictions(df: pd.DataFrame, preds: list[dict[str, Any]], latency_s: float) -> dict[str, Any]:
    pred_actions = pd.Series([p["selected_action"] for p in preds], name="predicted_action")

    for pred in preds:
        expected_no_action = pred.get("selected_action") == "do_nothing"
        if bool(pred.get("no_action")) != expected_no_action:
            raise ValueError("no_action consistency violation: no_action must be true iff action is do_nothing")

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
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        df["action_class"], pred_actions, labels=labels, zero_division=0
    )

    true_counts = df["action_class"].value_counts().to_dict()
    pred_counts = pred_actions.value_counts().to_dict()

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
        "action_class_distribution_true": true_counts,
        "action_class_distribution_pred": pred_counts,
        "prediction_diversity": {
            "predicted_action_entropy": _entropy_from_counts(pred_counts),
            "distribution_shift_l1": _l1_distribution_shift(true_counts, pred_counts, labels),
        },
        "no_action_distribution": {
            "true_no_action_rate": float((df["action_class"] == "do_nothing").mean()),
            "pred_no_action_rate": float((pred_actions == "do_nothing").mean()),
        },
        "multiclass": {
            "labels": labels,
            "accuracy": float(accuracy_score(df["action_class"], pred_actions)),
            "balanced_accuracy": float(balanced_accuracy_score(df["action_class"], pred_actions)),
            "macro_f1": float(f1_score(df["action_class"], pred_actions, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(df["action_class"], pred_actions, labels=labels, average="weighted", zero_division=0)
            ),
            "per_class": {
                label: {
                    "precision": float(per_precision[idx]),
                    "recall": float(per_recall[idx]),
                    "f1": float(per_f1[idx]),
                    "support": int(per_support[idx]),
                }
                for idx, label in enumerate(labels)
            },
            "confusion_matrix": multi_cm.tolist(),
            "per_action_true_counts": true_counts,
            "per_action_pred_counts": pred_counts,
        },
    }


def timed_decisions(fn, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    preds = [fn(r) for r in rows]
    return preds, time.perf_counter() - start
