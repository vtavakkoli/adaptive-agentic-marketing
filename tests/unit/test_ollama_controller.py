from __future__ import annotations

from typing import Any

from src.agentic import ollama_controller


class _FakeResponse:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"response": self.response_text, "context": list(range(1000)), "eval_count": 10}


class _FakeClient:
    response_text = "NOT JSON"
    last_payload: dict[str, Any] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def post(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        _FakeClient.last_payload = kwargs.get("json", {})
        return _FakeResponse(_FakeClient.response_text)


def test_decide_logs_parse_error_and_falls_back(monkeypatch) -> None:
    captured_events: list[str] = []

    def _capture_event(logger: Any, event: str, **payload: Any) -> None:
        captured_events.append(event)

    monkeypatch.setattr(ollama_controller, "log_event", _capture_event)
    monkeypatch.setattr(ollama_controller.httpx, "Client", _FakeClient)
    _FakeClient.response_text = "NOT JSON"

    client = ollama_controller.OllamaJSONClient("http://localhost:11434", "gemma4:e2b", retries=0)
    out = client.decide({"features": {}, "scores": {}})

    assert "llm_parse_error" in captured_events
    assert out["rationale"] == "fallback_due_to_slm_error"


def test_decide_parses_markdown_wrapped_json(monkeypatch) -> None:
    monkeypatch.setattr(ollama_controller.httpx, "Client", _FakeClient)
    _FakeClient.response_text = """```json
{
  "selected_action": "send_reminder",
  "confidence": 0.9,
  "no_action": 0,
  "rationale": "ok"
}
```"""

    client = ollama_controller.OllamaJSONClient("http://localhost:11434", "gemma4:e2b", retries=0)
    out = client.decide({"features": {}, "scores": {}})

    assert out["selected_action"] == "send_reminder"
    assert out["no_action"] is False
    assert out["confidence"] == 0.9


def test_decide_sanitizes_nan_in_prompt_payload(monkeypatch) -> None:
    monkeypatch.setattr(ollama_controller.httpx, "Client", _FakeClient)
    _FakeClient.response_text = """{"selected_action":"do_nothing","confidence":1,"no_action":true,"rationale":"ok"}"""

    client = ollama_controller.OllamaJSONClient("http://localhost:11434", "gemma4:e2b", retries=0)
    out = client.decide({"features": {"bucket_count": float("nan")}, "scores": {}})

    assert out["selected_action"] == "do_nothing"
    assert _FakeClient.last_payload is not None
    prompt = _FakeClient.last_payload["prompt"]
    assert "NaN" not in prompt


def test_decide_parses_escaped_json_string(monkeypatch) -> None:
    monkeypatch.setattr(ollama_controller.httpx, "Client", _FakeClient)
    _FakeClient.response_text = '{\\n  \\"selected_action\\": \\"do_nothing\\",\\n  \\"confidence\\": 0.7,\\n  \\"no_action\\": 1,\\n  \\"rationale\\": \\"escaped\\"\\n}'

    client = ollama_controller.OllamaJSONClient("http://localhost:11434", "gemma4:e2b", retries=0)
    out = client.decide({"features": {}, "scores": {}})

    assert out["selected_action"] == "do_nothing"
    assert out["no_action"] is True
    assert out["confidence"] == 0.7


def test_decide_parses_key_value_response(monkeypatch) -> None:
    monkeypatch.setattr(ollama_controller.httpx, "Client", _FakeClient)
    _FakeClient.response_text = (
        "selected_action:send_reminder,confidence:0.95,no_action:0.05,"
        "rationale:The customer has high need and relevance."
    )

    client = ollama_controller.OllamaJSONClient("http://localhost:11434", "gemma4:e2b", retries=0)
    out = client.decide({"features": {}, "scores": {}})

    assert out["selected_action"] == "send_reminder"
    assert out["confidence"] == 0.95
    assert out["no_action"] is False
