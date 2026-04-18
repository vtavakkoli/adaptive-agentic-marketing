from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.rl.action_mapping import ACTION_ID_TO_NAME
from src.rl.types import MarketingState, TransitionConfig


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _noise(rng: np.random.Generator, scale: float) -> float:
    return float(rng.normal(loc=0.0, scale=scale))


def apply_transition(state: MarketingState, action: int, cfg: TransitionConfig, rng: np.random.Generator, reminder_valid: bool) -> MarketingState:
    name = ACTION_ID_TO_NAME[action]
    next_state = state
    readiness = state.readiness_score
    prior_response = state.prior_response_rate
    fatigue = state.fatigue_score
    intrusion = state.intrusiveness_risk
    touches = state.campaign_touches_30d
    freq_7d = state.frequency_7d

    if name == "do_nothing":
        readiness = _clip01(readiness - cfg.do_nothing_opportunity_decay + _noise(rng, cfg.noise_scale))
        fatigue = _clip01(fatigue - cfg.do_nothing_fatigue_decay + _noise(rng, cfg.noise_scale))
        intrusion = _clip01(intrusion - cfg.do_nothing_fatigue_decay * 0.6 + _noise(rng, cfg.noise_scale))
        freq_7d = max(0.0, freq_7d - 1.0)
    elif name == "defer_action":
        readiness = _clip01(readiness - cfg.defer_conversion_decay + _noise(rng, cfg.noise_scale))
        fatigue = _clip01(fatigue - cfg.defer_fatigue_decay + _noise(rng, cfg.noise_scale))
        intrusion = _clip01(intrusion - cfg.defer_fatigue_decay * 0.8 + _noise(rng, cfg.noise_scale))
        freq_7d = max(0.0, freq_7d - 1.0)
    elif name == "send_information":
        gain = cfg.information_relevance_gain * state.offer_relevance
        prior_response = _clip01(prior_response + gain + _noise(rng, cfg.noise_scale))
        readiness = _clip01(readiness + gain * 0.8 + _noise(rng, cfg.noise_scale))
        fatigue = _clip01(fatigue + cfg.information_fatigue_increase + _noise(rng, cfg.noise_scale))
        intrusion = _clip01(intrusion + cfg.information_intrusion_increase + _noise(rng, cfg.noise_scale))
        touches += 1.0
        freq_7d = min(7.0, freq_7d + 1.0)
    else:  # send_reminder
        boost_base = cfg.reminder_boost * (0.7 * state.has_prior_offer_exposure + 0.3 * state.has_incomplete_intent)
        if reminder_valid:
            prior_response = _clip01(prior_response + boost_base + _noise(rng, cfg.noise_scale))
            readiness = _clip01(readiness + boost_base * 0.9 + _noise(rng, cfg.noise_scale))
        else:
            prior_response = _clip01(prior_response - cfg.reminder_misuse_penalty + _noise(rng, cfg.noise_scale))
            readiness = _clip01(readiness - cfg.reminder_misuse_penalty + _noise(rng, cfg.noise_scale))
        fatigue = _clip01(fatigue + cfg.reminder_fatigue_increase + _noise(rng, cfg.noise_scale))
        intrusion = _clip01(intrusion + cfg.reminder_intrusion_increase + _noise(rng, cfg.noise_scale))
        touches += 1.0
        freq_7d = min(7.0, freq_7d + 1.0)

    need = _clip01(0.55 * readiness + 0.45 * (1.0 - state.recency_days / 90.0))
    relevance = _clip01(0.6 * prior_response + 0.4 * need)

    return replace(
        next_state,
        recency_days=float(np.clip(state.recency_days + 1.0, 0.0, 365.0)),
        frequency_7d=float(np.clip(freq_7d, 0.0, 7.0)),
        campaign_touches_30d=float(np.clip(touches, 0.0, 30.0)),
        prior_response_rate=prior_response,
        need_score=need,
        fatigue_score=fatigue,
        intrusiveness_risk=intrusion,
        offer_relevance=relevance,
        readiness_score=readiness,
    )
