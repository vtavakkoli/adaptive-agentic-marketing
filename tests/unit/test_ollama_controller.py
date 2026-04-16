from __future__ import annotations

from typing import Any

from src.agentic import ollama_controller


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"response": "NOT JSON"}


class _FakeClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def post(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse()


def test_decide_logs_parse_error_and_falls_back(monkeypatch) -> None:
    captured_events: list[str] = []

    def _capture_event(logger: Any, event: str, **payload: Any) -> None:
        captured_events.append(event)

    monkeypatch.setattr(ollama_controller, "log_event", _capture_event)
    monkeypatch.setattr(ollama_controller.httpx, "Client", _FakeClient)

    client = ollama_controller.OllamaJSONClient("http://localhost:11434", "gemma4:e2b", retries=0)
    out = client.decide({"features": {}, "scores": {}})

    assert "llm_parse_error" in captured_events
    assert out["rationale"] == "fallback_due_to_slm_error"
