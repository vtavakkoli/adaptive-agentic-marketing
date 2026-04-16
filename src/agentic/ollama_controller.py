from __future__ import annotations

import json
from typing import Any

import httpx


class OllamaJSONClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 20, retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.retries = retries

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "Return strict JSON with keys: selected_action, confidence, no_action, rationale. "
            "Allowed actions: recommend_offer_a,recommend_offer_b,send_information,send_reminder,defer_action,do_nothing. "
            f"Input={json.dumps(payload)}"
        )
        for _ in range(self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(
                        f"{self.base_url}/api/generate",
                        json={"model": self.model, "prompt": prompt, "stream": False},
                    )
                    response.raise_for_status()
                    text = response.json().get("response", "{}")
                    parsed = json.loads(text)
                    if "selected_action" in parsed and "confidence" in parsed:
                        return parsed
            except Exception:
                continue
        return {
            "selected_action": "defer_action",
            "confidence": 0.5,
            "no_action": False,
            "rationale": "fallback_due_to_slm_error",
        }
