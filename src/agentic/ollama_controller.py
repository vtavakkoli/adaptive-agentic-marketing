from __future__ import annotations

import json
import math
from typing import Any

import httpx

from src.utils.logging_utils import configure_logging, log_event

ALLOWED_ACTIONS = {
    "recommend_offer_a",
    "recommend_offer_b",
    "send_information",
    "send_reminder",
    "defer_action",
    "do_nothing",
}


class OllamaJSONClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 90, retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.retries = retries
        self.logger = configure_logging()

    def _sanitize_for_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_for_json(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _compact_response_payload(self, response_payload: dict[str, Any]) -> dict[str, Any]:
        compact = {k: v for k, v in response_payload.items() if k not in {"context", "response"}}
        response_text = response_payload.get("response", "")
        compact["response_preview"] = response_text[:500] if isinstance(response_text, str) else ""
        return compact

    def _extract_json(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return stripped[start : end + 1]
        return stripped

    def _normalize_decision(self, parsed: dict[str, Any]) -> dict[str, Any]:
        action = str(parsed.get("selected_action", "defer_action"))
        if action not in ALLOWED_ACTIONS:
            action = "defer_action"

        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        raw_no_action = parsed.get("no_action", action == "do_nothing")
        if isinstance(raw_no_action, str):
            no_action = raw_no_action.strip().lower() in {"true", "1", "yes"}
        else:
            no_action = bool(raw_no_action)

        rationale = str(parsed.get("rationale", "")).strip() or "model_response"

        return {
            "selected_action": action,
            "confidence": confidence,
            "no_action": no_action,
            "rationale": rationale,
        }

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        safe_payload = self._sanitize_for_json(payload)
        prompt = (
            "Return strict JSON with keys: selected_action, confidence, no_action, rationale. "
            "Allowed actions: recommend_offer_a,recommend_offer_b,send_information,send_reminder,defer_action,do_nothing. "
            "Return only JSON without markdown code fences. "
            f"Input={json.dumps(safe_payload, allow_nan=False)}"
        )
        log_event(self.logger, "llm_request", model=self.model, payload=safe_payload, prompt=prompt)
        for _ in range(self.retries + 1):
            try:
                timeout = httpx.Timeout(connect=10.0, read=float(self.timeout_s), write=30.0, pool=10.0)
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        f"{self.base_url}/api/generate",
                        json={"model": self.model, "prompt": prompt, "stream": False},
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    text = response_json.get("response", "")
                    if not isinstance(text, str) or not text.strip():
                        log_event(
                            self.logger,
                            "llm_empty_response",
                            model=self.model,
                            response_payload=self._compact_response_payload(response_json),
                        )
                        continue
                    try:
                        parsed = json.loads(self._extract_json(text))
                    except json.JSONDecodeError as exc:
                        log_event(
                            self.logger,
                            "llm_parse_error",
                            model=self.model,
                            error=str(exc),
                            raw_response=text[:4000],
                            response_payload=self._compact_response_payload(response_json),
                        )
                        continue
                    normalized = self._normalize_decision(parsed)
                    log_event(self.logger, "llm_response", model=self.model, raw_response=text[:4000], parsed_response=normalized)
                    if "selected_action" in parsed and "confidence" in parsed:
                        return normalized
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
