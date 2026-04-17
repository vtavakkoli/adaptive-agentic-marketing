from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBuilderConfig:
    enable_reminder_features: bool = True
    enable_defer_features: bool = True
    enable_temporal_features: bool = True


def _safe_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_features(df: pd.DataFrame, cfg: FeatureBuilderConfig | None = None) -> pd.DataFrame:
    cfg = cfg or FeatureBuilderConfig()
    out = df.copy()

    recency = _safe_col(out, "recency_days", 30.0)
    prior_response = _safe_col(out, "prior_response_rate", 0.5).clip(0, 1)
    freq_7d = _safe_col(out, "frequency_7d", 0.0)
    touches_30d = _safe_col(out, "campaign_touches_30d", 0.0)
    fatigue = _safe_col(out, "fatigue_score", (touches_30d / 12.0).clip(0, 1)).clip(0, 1)

    if cfg.enable_temporal_features:
        out["touches_last_7d"] = freq_7d
        out["touches_last_14d"] = (freq_7d * 1.7).round(0)
        out["touches_last_30d"] = touches_30d.clip(lower=freq_7d)
        out["days_since_last_contact"] = recency
        out["days_since_last_positive_response"] = (1.0 - prior_response) * 30.0
        out["offer_age_days"] = _safe_col(out, "offer_age_days", np.minimum(recency + 5, 120))
        out["cadence_bucket"] = pd.cut(freq_7d, bins=[-np.inf, 1, 3, 6, np.inf], labels=[0, 1, 2, 3]).astype(int)

    if cfg.enable_reminder_features:
        offer_id = out.get("offer_id", pd.Series("", index=out.index)).astype(str)
        out["has_prior_offer_exposure"] = ((recency < 45) | offer_id.str.contains("offer|reminder", case=False)).astype(int)
        out["has_prior_engagement_on_offer"] = (prior_response > 0.45).astype(int)
        out["has_incomplete_intent"] = ((prior_response > 0.25) & (prior_response < 0.65)).astype(int)
        out["days_since_last_offer_touch"] = np.maximum(recency - 3, 0)
        out["num_prior_reminders"] = np.where(offer_id.str.contains("reminder", case=False), 1, 0) + (freq_7d > 2).astype(int)
        out["last_engagement_type"] = np.where(prior_response > 0.65, "clicked", np.where(prior_response > 0.35, "opened", "none"))
        out["campaign_stage"] = np.where(recency < 10, "early", np.where(recency < 35, "mid", "late"))
        out["abandoned_action_flag"] = ((prior_response > 0.3) & (freq_7d < 2)).astype(int)

    if cfg.enable_defer_features:
        out["cooldown_remaining_days"] = np.maximum(7 - recency, 0)
        out["best_contact_window_score"] = (1.0 - fatigue).clip(0, 1)
        out["readiness_score"] = (0.55 * (1 - fatigue) + 0.45 * prior_response).clip(0, 1)
        out["temporary_suppression_flag"] = (fatigue > 0.75).astype(int)
        out["days_since_last_meaningful_engagement"] = np.maximum((1 - prior_response) * 21, 0)
        out["fatigue_margin"] = (0.7 - fatigue).clip(-1, 1)
        out["timing_mismatch_score"] = (np.abs(recency - 14) / 14).clip(0, 2)

    return out
