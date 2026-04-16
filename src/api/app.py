from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.agentic.controller import AdaptiveAgenticController
from src.config import load_yaml
from src.schemas import DecisionRequest, DecisionResponse, HealthResponse

app = FastAPI(title="adaptive-agentic-marketing")
CFG_PATH = Path("configs/default.yaml")
cfg = load_yaml(CFG_PATH)
controller = AdaptiveAgenticController(cfg)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=controller.model_loaded,
        ollama_enabled=controller.slm_enabled,
    )


@app.get("/config")
def get_config() -> dict:
    return cfg


@app.post("/decide", response_model=DecisionResponse)
def decide(payload: DecisionRequest) -> DecisionResponse:
    features = payload.model_dump()
    features.setdefault("need_score", 0.5)
    features.setdefault("fatigue_score", min(features["frequency_7d"] / 7.0, 1.0))
    features.setdefault("intrusiveness_risk", features["fatigue_score"])
    features.setdefault("offer_relevance", 0.5)
    features.setdefault("prior_response_rate", 0.5)
    out = controller.decide(features)
    return DecisionResponse(**{k: out[k] for k in DecisionResponse.model_fields})


@app.get("/report/latest")
def latest_report() -> dict:
    report = Path("outputs/reports/metrics.json")
    if not report.exists():
        raise HTTPException(status_code=404, detail="No report found")
    return json.loads(report.read_text(encoding="utf-8"))
