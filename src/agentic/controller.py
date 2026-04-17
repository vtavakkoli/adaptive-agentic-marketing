from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.agentic.ollama_controller import OllamaJSONClient
from src.content.generation_module import generate_message
from src.data.feature_builder import FeatureBuilderConfig, build_features
from src.explanations.explanation_module import build_explanation
from src.models.xgboost_module import XGBoostModule
from src.policy.controller import HierarchicalPolicyController
from src.policy.guardrails import evaluate_guardrails
from src.policy.rules_engine import apply_rules

FINAL_ACTIONS = {
    "do_nothing",
    "defer_action",
    "send_information",
    "send_reminder",
}

LLM_ALLOWED_FEATURE_KEYS = {
    "recency_days",
    "frequency_7d",
    "avg_basket_value",
    "campaign_touches_30d",
    "prior_response_rate",
    "need_score",
    "fatigue_score",
    "intrusiveness_risk",
    "offer_relevance",
    "channel",
    "offer_id",
    "edge_case_flag",
}

LEAKAGE_BLOCKLIST = {
    "action_class",
    "no_action_preferred",
    "source_case_id",
    "target",
    "label",
    "ground_truth",
    "y_true",
}

MODE_ALIASES = {
    "adaptive_full": "adaptive_simple",
    "adaptive_full_framework": "adaptive_simple",
}


class AdaptiveAgenticController:
    def __init__(self, policy_cfg: dict[str, Any]) -> None:
        self.policy_cfg = policy_cfg
        self.xgb = XGBoostModule()
        model_path = Path(policy_cfg.get("model_path", "outputs/models/xgboost.joblib"))
        self.model_loaded = model_path.exists()
        if self.model_loaded:
            self.xgb.load(model_path)
        self.slm_enabled = policy_cfg.get("slm", {}).get("enabled", True)
        self.ollama = OllamaJSONClient(
            base_url=os.getenv("OLLAMA_BASE_URL", policy_cfg.get("slm", {}).get("base_url", "http://host.docker.internal:11434")),
            model=policy_cfg.get("slm", {}).get("model", "gemma4:e2b"),
            timeout_s=int(policy_cfg.get("slm", {}).get("timeout_s", 90)),
            retries=int(policy_cfg.get("slm", {}).get("retries", 2)),
        )
        self.hierarchical = HierarchicalPolicyController(policy_cfg)

    def _sanitize_features_for_llm(self, features: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in features.items():
            lowered = key.lower()
            if key in LEAKAGE_BLOCKLIST or any(token in lowered for token in LEAKAGE_BLOCKLIST):
                continue
            if key in LLM_ALLOWED_FEATURE_KEYS or key.endswith("_bucket"):
                sanitized[key] = value
        return sanitized

    def _build_llm_payload(self, features: dict[str, Any], scores: dict[str, float]) -> dict[str, Any]:
        return {
            "features": self._sanitize_features_for_llm(features),
            "scores": scores,
        }

    def _enforce_no_action_consistency(self, action: str, no_action: bool) -> bool:
        expected = action == "do_nothing"
        if no_action != expected:
            return expected
        return no_action

    def _decide_simple(self, features: dict[str, Any], mode: str, rule, scores: dict[str, float]) -> tuple[str, float]:
        action = "defer_action"
        confidence = 0.6

        if self.model_loaded and mode not in {"rules_only", "ablation_no_xgboost"}:
            pred = self.xgb.predict_scores(pd.DataFrame([features])).iloc[0].to_dict()
            scores.update({k: float(v) for k, v in pred.items()})

        if mode != "ablation_no_rules" and rule.action == "do_nothing":
            action = "do_nothing"
            confidence = 0.95
        elif mode == "xgboost_only":
            action = "do_nothing" if scores["contact_risk_pred"] > 0.65 else "send_information"
            confidence = 0.75
        elif mode == "slm_only" and self.slm_enabled:
            slm = self.ollama.decide(self._build_llm_payload(features, scores))
            action = slm.get("selected_action", "defer_action")
            confidence = float(slm.get("confidence", 0.55))
        else:
            if self.slm_enabled and mode != "ablation_no_content_generation":
                slm = self.ollama.decide(self._build_llm_payload(features, scores))
                action = slm.get("selected_action", "defer_action")
                confidence = float(slm.get("confidence", 0.65))
            else:
                action = "do_nothing" if scores["contact_risk_pred"] > 0.7 else "defer_action"
        return action, confidence

    def _decide_hierarchical(self, features: dict[str, Any], scores: dict[str, float]) -> dict[str, Any]:
        fe_df = build_features(pd.DataFrame([features]), FeatureBuilderConfig())
        row = fe_df.iloc[0].to_dict()
        fatigue = float(row.get("fatigue_score", 0.0))
        risk = float(row.get("intrusiveness_risk", 0.0))
        readiness = float(row.get("readiness_score", 0.5))

        p_action = float((0.58 * (1 - fatigue) + 0.42 * readiness) * (1 - 0.35 * risk))
        p_action = max(0.0, min(1.0, p_action))
        stage_a = {"do_nothing": 1 - p_action, "action": p_action}

        p_reminder = float(0.50 * row.get("has_prior_offer_exposure", 0) + 0.35 * row.get("has_incomplete_intent", 0) + 0.15 * row.get("abandoned_action_flag", 0))
        p_defer = float(0.55 * fatigue + 0.45 * (1 - readiness))
        p_info = float(max(0.0, 1 - (p_reminder + p_defer) / 1.4))
        total = p_reminder + p_defer + p_info + 1e-9
        stage_b = {
            "defer_action": p_defer / total,
            "send_information": p_info / total,
            "send_reminder": p_reminder / total,
        }

        decision = self.hierarchical.decide(row, stage_a, stage_b)
        scores.update(
            {
                "stage_a_p_action": float(stage_a["action"]),
                "stage_b_p_defer_action": float(stage_b["defer_action"]),
                "stage_b_p_send_information": float(stage_b["send_information"]),
                "stage_b_p_send_reminder": float(stage_b["send_reminder"]),
            }
        )
        return {
            "selected_action": decision.final_action,
            "confidence": decision.calibrated_confidence,
            "stage_a_prediction": decision.stage_a_prediction,
            "stage_a_probability": decision.stage_a_probability_action,
            "stage_b_prediction": decision.stage_b_prediction,
            "stage_b_probability": decision.stage_b_probability,
            "guardrail_overrode": decision.guardrail_overrode,
            "fallback_reason": decision.fallback_reason,
            "decision_trace": decision.trace,
            "top_features": decision.top_features,
        }

    def decide(self, features: dict[str, Any], mode: str = "adaptive_simple") -> dict[str, Any]:
        mode = MODE_ALIASES.get(mode, mode)
        rule = apply_rules(features, self.policy_cfg.get("rules", {}))
        scores = {
            "contact_risk_pred": float(features.get("fatigue_score", 0.0)),
            "response_propensity_pred": float(features.get("prior_response_rate", 0.5)),
            "fatigue_risk_proxy": float(features.get("fatigue_score", 0.0)),
            "intrusion_risk_proxy": float(features.get("intrusiveness_risk", 0.0)),
            "offer_relevance_proxy": float(features.get("offer_relevance", 0.5)),
        }

        if mode == "adaptive_hierarchical":
            out = self._decide_hierarchical(features, scores)
            action = out["selected_action"]
            confidence = float(out["confidence"])
        else:
            action, confidence = self._decide_simple(features, mode, rule, scores)
            out = {}

        if action not in FINAL_ACTIONS:
            action = "defer_action"

        # keep hard guardrail guarantee as final enforcement
        guard = evaluate_guardrails(build_features(pd.DataFrame([features])).iloc[0].to_dict(), self.policy_cfg.get("guardrails", {}))
        if action not in guard.allowed_actions:
            action = "do_nothing" if "do_nothing" in guard.allowed_actions else "defer_action"
            out["guardrail_overrode"] = True
            out["fallback_reason"] = out.get("fallback_reason") or "final_guardrail_enforcement"

        no_action = self._enforce_no_action_consistency(action, action == "do_nothing")
        explanation = build_explanation(action, scores, rule_reason=rule.reason)

        content = ""
        if mode != "ablation_no_content_generation":
            content = generate_message(action)

        return {
            "selected_action": action,
            "confidence": confidence,
            "explanation": explanation,
            "no_action": no_action,
            "supporting_scores": scores,
            "content": content,
            "rule_reason": rule.reason,
            "rule_forced": rule.violated,
            "mode_used": mode,
            **out,
        }
