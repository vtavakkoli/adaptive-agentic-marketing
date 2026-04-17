from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Action = Literal[
    "send_information",
    "send_reminder",
    "defer_action",
    "do_nothing",
]


class DecisionRequest(BaseModel):
    customer_id: str
    recency_days: int = Field(ge=0)
    frequency_7d: int = Field(ge=0)
    avg_basket_value: float = Field(ge=0)
    offer_id: str = "offer_a"
    channel: str = "email"


class DecisionResponse(BaseModel):
    selected_action: Action
    confidence: float
    explanation: str
    no_action: bool
    supporting_scores: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ollama_enabled: bool
