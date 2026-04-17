from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.policy.guardrails import GuardrailResult, evaluate_guardrails


@dataclass
class ControllerDecision:
    final_action: str
    calibrated_confidence: float
    stage_a_prediction: str
    stage_a_probability_action: float
    stage_b_prediction: str | None
    stage_b_probability: float | None
    guardrail_overrode: bool
    fallback_reason: str | None
    top_features: list[str]
    trace: dict[str, Any]


class HierarchicalPolicyController:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def decide(
        self,
        features: dict[str, Any],
        stage_a_probs: dict[str, float],
        stage_b_probs: dict[str, float] | None,
        ood_score: float | None = None,
    ) -> ControllerDecision:
        thresholds = self.cfg.get("controller", {})
        risk = float(features.get("intrusiveness_risk", 0.0))
        action_prob = float(stage_a_probs.get("action", 0.0))
        no_action_prob = float(stage_a_probs.get("do_nothing", 1.0 - action_prob))
        guardrails: GuardrailResult = evaluate_guardrails(features, self.cfg.get("guardrails", {}))

        risk_abstain = float(thresholds.get("risk_abstain_threshold", 0.7))
        if no_action_prob >= float(thresholds.get("binary_action_threshold", 0.52)) or risk >= risk_abstain:
            return self._wrap("do_nothing", no_action_prob, stage_a_probs, None, guardrails, "high_risk_or_binary_no_action", features)

        if stage_b_probs is None:
            return self._wrap("defer_action", action_prob, stage_a_probs, None, guardrails, "missing_stage_b", features)

        allowed_stage_b = {k: v for k, v in stage_b_probs.items() if k in guardrails.allowed_actions}
        if not allowed_stage_b:
            return self._wrap("do_nothing", no_action_prob, stage_a_probs, stage_b_probs, guardrails, "guardrail_removed_all_actions", features)

        ranked = sorted(allowed_stage_b.items(), key=lambda kv: kv[1], reverse=True)
        top_action, top_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        gap = top_prob - second_prob

        min_action_conf = float(thresholds.get("min_action_confidence", 0.45))
        top2_gap_th = float(thresholds.get("top2_gap_threshold", 0.08))
        reminder_min = float(thresholds.get("min_reminder_confidence", 0.52))
        if top_action == "send_reminder" and top_prob < reminder_min:
            return self._wrap("defer_action", top_prob, stage_a_probs, stage_b_probs, guardrails, "reminder_confidence_too_low", features)

        if top_prob < min_action_conf or gap < top2_gap_th or (ood_score is not None and ood_score > 0.8):
            fallback = "defer_action" if "defer_action" in guardrails.allowed_actions and risk < risk_abstain else "do_nothing"
            return self._wrap(fallback, max(top_prob, no_action_prob), stage_a_probs, stage_b_probs, guardrails, "uncertainty_fallback", features)

        return self._wrap(top_action, top_prob, stage_a_probs, stage_b_probs, guardrails, None, features)

    def _wrap(
        self,
        action: str,
        conf: float,
        stage_a: dict[str, float],
        stage_b: dict[str, float] | None,
        guardrails: GuardrailResult,
        fallback_reason: str | None,
        features: dict[str, Any],
    ) -> ControllerDecision:
        stage_b_pred = None
        stage_b_prob = None
        if stage_b:
            stage_b_pred, stage_b_prob = max(stage_b.items(), key=lambda x: x[1])
        top_features = sorted(
            ["fatigue_score", "intrusiveness_risk", "readiness_score", "has_prior_offer_exposure", "has_incomplete_intent"],
            key=lambda k: abs(float(features.get(k, 0.0))),
            reverse=True,
        )[:4]
        trace = {
            "guardrail_blocked": guardrails.blocked_reasons,
            "stage_a_probs": stage_a,
            "stage_b_probs": stage_b or {},
            "fallback_reason": fallback_reason,
        }
        return ControllerDecision(
            final_action=action,
            calibrated_confidence=float(conf),
            stage_a_prediction="action" if stage_a.get("action", 0) >= stage_a.get("do_nothing", 0) else "do_nothing",
            stage_a_probability_action=float(stage_a.get("action", 0.0)),
            stage_b_prediction=stage_b_pred,
            stage_b_probability=float(stage_b_prob) if stage_b_prob is not None else None,
            guardrail_overrode=action not in (stage_b or {action: 1.0}),
            fallback_reason=fallback_reason,
            top_features=top_features,
            trace=trace,
        )
