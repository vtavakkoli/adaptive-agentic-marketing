from __future__ import annotations

import json
import math
import re
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


def maybe_unescape(text: str) -> str:
    """
    Decode escaped sequences safely, avoiding corruption of valid JSON and UTF-8 characters.
    It only unescapes if the string is thoroughly over-escaped (e.g., literal \\n and \\").
    """
    if not isinstance(text, str):
        return text

    text = text.strip()

    # If wrapped completely in a JSON string literal (e.g., "\"{\\n...}\"")
    if text.startswith('"') and text.endswith('"'):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, str):
                text = parsed
        except Exception:
            pass

    # If the text has literal backslash-quotes and NO real quotes, it is fully over-escaped.
    if '\\"' in text and '"' not in text.replace('\\"', ''):
        text = text.replace('\\"', '"').replace('\\n', '\n')

    return text


def strip_code_fences(text: str) -> str:
    """
    Remove Markdown code fences such as ```json ... ``` and return
    the inner JSON text without truncating nested braces.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback to remove malformed backticks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def normalize_no_action(value: Any, selected_action: str | None = None) -> bool:
    """
    Normalize no_action to a strict boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value >= 0.5

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False

    if selected_action is not None:
        return selected_action == "do_nothing"

    return False


def parse_raw_response(raw_response: str) -> dict[str, Any]:
    """
    Parse an LLM response that may contain:
    - escaped characters like \\n and \\
    - fenced markdown ```json ... ```
    - non-boolean no_action values
    """
    text = raw_response.strip()

    text = strip_code_fences(text)
    text = maybe_unescape(text)

    # Clean subset of string by finding outer-most braces
    if not text.lstrip().startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as json_exc:
        # Fallback parser for non-JSON outputs such as:
        # selected_action:send_reminder,confidence:0.95,no_action:0.05,rationale:...
        parts = dict(
            (
                match.group(1).strip(),
                match.group(2).strip(),
            )
            for match in re.finditer(
                r"(selected_action|confidence|no_action|rationale)\s*:\s*(.*?)(?=,\s*(?:selected_action|confidence|no_action|rationale)\s*:|$)",
                text,
                re.DOTALL,
            )
        )
        if {"selected_action", "confidence"}.issubset(parts):
            return parts
        raise json_exc


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
        compact = {
            k: v
            for k, v in response_payload.items()
            if k
            in {
                "model",
                "created_at",
                "done",
                "done_reason",
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            }
        }
        response_text = response_payload.get("response", "")
        compact["response_preview"] = response_text[:300] if isinstance(response_text, str) else ""
        return compact

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
        no_action = normalize_no_action(raw_no_action, selected_action=action)

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

        features = safe_payload.get("features", {}) if isinstance(safe_payload, dict) else {}
        scores = safe_payload.get("scores", {}) if isinstance(safe_payload, dict) else {}
        log_event(
            self.logger,
            "llm_request",
            model=self.model,
            case_id=features.get("case_id") if isinstance(features, dict) else None,
            features_keys=sorted(features.keys()) if isinstance(features, dict) else [],
            scores_keys=sorted(scores.keys()) if isinstance(scores, dict) else [],
        )

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "format": "json",
                        },
                    )
                    response.raise_for_status()
                    response_payload = response.json()

                log_event(
                    self.logger,
                    "llm_response",
                    model=self.model,
                    case_id=features.get("case_id") if isinstance(features, dict) else None,
                    attempt=attempt,
                    payload=self._compact_response_payload(response_payload),
                )

                parsed = parse_raw_response(str(response_payload.get("response", "")).strip())
                return self._normalize_decision(parsed)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                log_event(
                    self.logger,
                    "llm_parse_error",
                    model=self.model,
                    case_id=features.get("case_id") if isinstance(features, dict) else None,
                    attempt=attempt,
                    error=str(exc),
                )
            except httpx.HTTPError as exc:
                last_error = exc
                log_event(
                    self.logger,
                    "llm_http_error",
                    model=self.model,
                    case_id=features.get("case_id") if isinstance(features, dict) else None,
                    attempt=attempt,
                    error=str(exc),
                )

        log_event(
            self.logger,
            "llm_fallback_decision",
            model=self.model,
            case_id=features.get("case_id") if isinstance(features, dict) else None,
            error=str(last_error) if last_error else "unknown_error",
        )
        return {
            "selected_action": "defer_action",
            "confidence": 0.0,
            "no_action": False,
            "rationale": "fallback_due_to_slm_error",
        }
