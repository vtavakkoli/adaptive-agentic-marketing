from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


ACTIONS = ["do_nothing", "defer_action", "send_information", "send_reminder"]


@dataclass(frozen=True)
class LabelDefinition:
    reminder_requires_exposure: bool = True
    reminder_requires_engagement_or_intent: bool = True


def reminder_prerequisites_met(row: pd.Series) -> bool:
    exposure = bool(row.get("has_prior_offer_exposure", 0))
    engaged = bool(row.get("has_prior_engagement_on_offer", 0) or row.get("has_incomplete_intent", 0) or row.get("abandoned_action_flag", 0))
    return exposure and engaged


def run_label_audit(df: pd.DataFrame, definition: LabelDefinition | None = None) -> dict[str, Any]:
    definition = definition or LabelDefinition()
    action = df.get("action_class", pd.Series("", index=df.index)).astype(str)

    reminder_labeled = df[action == "send_reminder"] if "action_class" in df.columns else df.iloc[0:0]
    missing_prereq = 0
    if not reminder_labeled.empty:
        missing_prereq = int((~reminder_labeled.apply(reminder_prerequisites_met, axis=1)).sum())

    multi_match = 0
    ambiguous = 0
    if {"no_action_preferred", "fatigue_score", "offer_relevance", "readiness_score"}.intersection(df.columns):
        is_do_nothing = df.get("no_action_preferred", 0).astype(bool)
        is_defer = (df.get("fatigue_score", 0.0) > 0.55) | (df.get("readiness_score", 0.5) < 0.4)
        is_info = df.get("offer_relevance", 0.0) > 0.4
        is_reminder = df.apply(reminder_prerequisites_met, axis=1) if len(df) else pd.Series(dtype=bool)
        stack = pd.concat([is_do_nothing, is_defer, is_info, is_reminder], axis=1).astype(int)
        matches = stack.sum(axis=1)
        multi_match = int((matches > 1).sum())
        ambiguous = int((matches == 0).sum())

    return {
        "rows": int(len(df)),
        "label_distribution": action.value_counts().to_dict(),
        "ambiguous_rows": ambiguous,
        "multi_class_match_rows": multi_match,
        "send_reminder_missing_prerequisites": missing_prereq,
    }


def feature_audit(df: pd.DataFrame, pred_df: pd.DataFrame | None = None) -> dict[str, Any]:
    target = df["action_class"].astype(str)

    def _mean_delta(a: str, b: str) -> dict[str, float]:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        left = df[target == a][num_cols].mean(numeric_only=True)
        right = df[target == b][num_cols].mean(numeric_only=True)
        delta = (left - right).abs().sort_values(ascending=False).head(8)
        return {k: float(v) for k, v in delta.items()}

    out = {
        "send_reminder_vs_send_information": _mean_delta("send_reminder", "send_information"),
        "defer_action_vs_send_information": _mean_delta("defer_action", "send_information"),
    }
    if pred_df is not None and {"selected_action"}.issubset(pred_df.columns):
        merged = df.copy()
        merged["pred"] = pred_df["selected_action"].astype(str).values
        for true_label, wrong in [("send_reminder", "send_information"), ("defer_action", "send_information")]:
            slice_df = merged[(merged["action_class"] == true_label) & (merged["pred"] == wrong)]
            out[f"misclassified_{true_label}_to_{wrong}"] = {
                "count": int(len(slice_df)),
                "avg_confidence": float(pd.to_numeric(pred_df.loc[slice_df.index, "confidence"], errors="coerce").fillna(0).mean())
                if len(slice_df)
                else 0.0,
            }
    return out
