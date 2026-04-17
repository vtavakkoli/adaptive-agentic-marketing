from __future__ import annotations

from typing import Any

import pandas as pd

from src.agentic.controller import AdaptiveAgenticController, FINAL_ACTIONS
from src.features.label_engineering import ACTION_ORDER


class _StubOllama:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = payload
        return {
            "selected_action": "send_information",
            "confidence": 0.7,
            "no_action": True,
            "rationale": "stub",
        }


class _StubXGB:
    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([{"contact_risk_pred": 0.1, "response_propensity_pred": 0.8}])


def _base_features() -> dict[str, Any]:
    return {
        "recency_days": 5,
        "frequency_7d": 1,
        "avg_basket_value": 20.0,
        "campaign_touches_30d": 2,
        "prior_response_rate": 0.4,
        "need_score": 0.7,
        "fatigue_score": 0.2,
        "intrusiveness_risk": 0.2,
        "offer_relevance": 0.7,
        "channel": "email",
        "offer_id": "offer_a",
        "edge_case_flag": False,
        "action_class": "send_information",
        "no_action_preferred": 0,
        "source_case_id": "abc",
    }


def test_llm_payload_removes_leaked_keys() -> None:
    controller = AdaptiveAgenticController({"slm": {"enabled": True}, "rules": {}})
    controller.model_loaded = True
    controller.xgb = _StubXGB()
    stub = _StubOllama()
    controller.ollama = stub

    controller.decide(_base_features(), mode="slm_only")

    assert stub.payload is not None
    llm_features = stub.payload["features"]
    assert "action_class" not in llm_features
    assert "no_action_preferred" not in llm_features
    assert "source_case_id" not in llm_features


def test_action_set_unified() -> None:
    assert set(ACTION_ORDER) == FINAL_ACTIONS


def test_no_action_consistency_enforced() -> None:
    controller = AdaptiveAgenticController({"slm": {"enabled": True}, "rules": {}})
    controller.model_loaded = True
    controller.xgb = _StubXGB()
    controller.ollama = _StubOllama()

    out = controller.decide(_base_features(), mode="slm_only")
    assert out["selected_action"] == "send_information"
    assert out["no_action"] is False
