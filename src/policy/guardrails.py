from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GuardrailResult:
    allowed_actions: set[str]
    blocked_reasons: dict[str, str]


def _reminder_prereq(features: dict[str, Any]) -> bool:
    return bool(features.get("has_prior_offer_exposure", 0)) and bool(
        features.get("has_prior_engagement_on_offer", 0)
        or features.get("has_incomplete_intent", 0)
        or features.get("abandoned_action_flag", 0)
    )


def evaluate_guardrails(features: dict[str, Any], cfg: dict[str, Any]) -> GuardrailResult:
    allowed = {"do_nothing", "defer_action", "send_information", "send_reminder"}
    blocked: dict[str, str] = {}

    fatigue_th = float(cfg.get("fatigue_threshold", 0.78))
    intrusion_th = float(cfg.get("intrusion_threshold", 0.72))
    max_touches_7d = int(cfg.get("max_touches_7d", 5))
    blocked_channels = set(cfg.get("blocked_channels", []))
    blocked_offers = set(cfg.get("policy_excluded_offers", []))

    if float(features.get("fatigue_score", 0.0)) >= fatigue_th or int(features.get("frequency_7d", 0)) >= max_touches_7d:
        allowed -= {"send_information", "send_reminder"}
        blocked["fatigue/contact_pressure"] = "fatigue_or_contact_pressure"
    if float(features.get("intrusiveness_risk", 0.0)) >= intrusion_th:
        allowed -= {"send_information", "send_reminder"}
        blocked["intrusion_risk"] = "intrusion_risk_high"
    if str(features.get("channel", "")) in blocked_channels:
        allowed -= {"send_information", "send_reminder"}
        blocked["channel_restriction"] = "channel_restricted"
    if str(features.get("offer_id", "")) in blocked_offers:
        allowed -= {"send_information", "send_reminder"}
        blocked["policy_exclusion"] = "offer_excluded"

    if not _reminder_prereq(features):
        allowed.discard("send_reminder")
        blocked["missing_prerequisites_for_reminder"] = "no_prior_exposure_or_intent"

    if not allowed:
        allowed.add("do_nothing")
    return GuardrailResult(allowed_actions=allowed, blocked_reasons=blocked)
