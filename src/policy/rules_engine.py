from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuleDecision:
    action: str
    reason: str
    violated: bool = False


def apply_rules(features: dict[str, Any], policy_cfg: dict[str, Any]) -> RuleDecision:
    freq_cap = policy_cfg.get("frequency_cap_7d", 5)
    fatigue_threshold = policy_cfg.get("fatigue_do_nothing_threshold", 0.7)
    relevance_threshold = policy_cfg.get("min_relevance_threshold", 0.35)

    if float(features.get("frequency_7d", 0)) >= freq_cap:
        return RuleDecision("do_nothing", "frequency_cap_triggered", True)
    if float(features.get("fatigue_score", 0.0)) >= fatigue_threshold:
        return RuleDecision("do_nothing", "high_fatigue", True)
    if float(features.get("offer_relevance", 0.0)) <= relevance_threshold:
        return RuleDecision("do_nothing", "low_relevance", True)
    if features.get("offer_id") in set(policy_cfg.get("invalid_offers", [])):
        return RuleDecision("do_nothing", "invalid_offer", True)
    return RuleDecision("defer_action", "no_rule_forced", False)
