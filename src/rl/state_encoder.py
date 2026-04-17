from __future__ import annotations

import hashlib

import numpy as np

from src.rl.types import MarketingState


def _stable_hash_norm(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    as_int = int(digest[:8], 16)
    return float(as_int / 0xFFFFFFFF)


def encode_state(state: MarketingState) -> np.ndarray:
    return np.asarray(
        [
            state.recency_days / 90.0,
            state.frequency_7d / 7.0,
            state.avg_basket_value / 250.0,
            state.campaign_touches_30d / 30.0,
            state.prior_response_rate,
            state.need_score,
            state.fatigue_score,
            state.intrusiveness_risk,
            state.offer_relevance,
            _stable_hash_norm(state.channel),
            _stable_hash_norm(state.offer_id),
            state.readiness_score,
            float(state.has_prior_offer_exposure),
            float(state.has_prior_engagement_on_offer),
            float(state.has_incomplete_intent),
            float(state.abandoned_action_flag),
        ],
        dtype=np.float32,
    )
