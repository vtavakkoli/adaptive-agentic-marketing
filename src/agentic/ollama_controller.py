from __future__ import annotations

import json
from typing import Any

import httpx

from src.utils.logging_utils import configure_logging, log_event


class OllamaJSONClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 90, retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.retries = retries
        self.logger = configure_logging()

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "Return strict JSON with keys: selected_action, confidence, no_action, rationale. "
            "Allowed actions: recommend_offer_a,recommend_offer_b,send_information,send_reminder,defer_action,do_nothing. "
            f"Input={json.dumps(payload)}"
        )
        log_event(self.logger, "llm_request", model=self.model, payload=payload, prompt=prompt)
        for _ in range(self.retries + 1):
            try:
                timeout = httpx.Timeout(connect=10.0, read=float(self.timeout_s), write=30.0, pool=10.0)
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        f"{self.base_url}/api/generate",
                        json={"model": self.model, "prompt": prompt, "stream": False},
                    )
                    response.raise_for_status()
                    text = response.json().get("response", "{}")
                    parsed = json.loads(text)
                    log_event(self.logger, "llm_response", model=self.model, raw_response=text, parsed_response=parsed)
                    if "selected_action" in parsed and "confidence" in parsed:
                        return parsed
            except Exception as exc:
                log_event(self.logger, "llm_error", model=self.model, error=str(exc))
                continue
        fallback = {
            "selected_action": "defer_action",
            "confidence": 0.5,
            "no_action": False,
            "rationale": "fallback_due_to_slm_error",
        }
        log_event(self.logger, "llm_fallback_response", model=self.model, parsed_response=fallback)
        return fallback
