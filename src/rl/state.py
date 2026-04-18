from __future__ import annotations

from typing import Any

from src.rl.types import MarketingState


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def state_from_row(row: dict[str, Any]) -> MarketingState:
    return MarketingState(
        recency_days=float(max(0.0, row.get("recency_days", 30.0))),
        frequency_7d=float(max(0.0, row.get("frequency_7d", 0.0))),
        avg_basket_value=float(max(0.0, row.get("avg_basket_value", 0.0))),
        campaign_touches_30d=float(max(0.0, row.get("campaign_touches_30d", 0.0))),
        prior_response_rate=_clip01(float(row.get("prior_response_rate", 0.5))),
        need_score=_clip01(float(row.get("need_score", 0.5))),
        fatigue_score=_clip01(float(row.get("fatigue_score", 0.0))),
        intrusiveness_risk=_clip01(float(row.get("intrusiveness_risk", 0.0))),
        offer_relevance=_clip01(float(row.get("offer_relevance", 0.5))),
        channel=str(row.get("channel", "unknown")),
        offer_id=str(row.get("offer_id", "unknown")),
        readiness_score=_clip01(float(row.get("readiness_score", 0.5))),
        has_prior_offer_exposure=int(bool(row.get("has_prior_offer_exposure", 0))),
        has_prior_engagement_on_offer=int(bool(row.get("has_prior_engagement_on_offer", 0))),
        has_incomplete_intent=int(bool(row.get("has_incomplete_intent", 0))),
        abandoned_action_flag=int(bool(row.get("abandoned_action_flag", 0))),
    )
