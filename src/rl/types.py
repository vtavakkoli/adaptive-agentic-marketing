from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MarketingState:
    recency_days: float
    frequency_7d: float
    avg_basket_value: float
    campaign_touches_30d: float
    prior_response_rate: float
    need_score: float
    fatigue_score: float
    intrusiveness_risk: float
    offer_relevance: float
    channel: str
    offer_id: str
    readiness_score: float
    has_prior_offer_exposure: int
    has_prior_engagement_on_offer: int
    has_incomplete_intent: int
    abandoned_action_flag: int


@dataclass(frozen=True)
class RewardBreakdown:
    conversion_reward: float = 0.0
    engagement_reward: float = 0.0
    relevance_reward: float = 0.0
    fatigue_penalty: float = 0.0
    intrusiveness_penalty: float = 0.0
    over_contact_penalty: float = 0.0
    illegal_action_penalty: float = 0.0
    unnecessary_action_penalty: float = 0.0
    abstention_reward: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.conversion_reward
            + self.engagement_reward
            + self.relevance_reward
            + self.abstention_reward
            - self.fatigue_penalty
            - self.intrusiveness_penalty
            - self.over_contact_penalty
            - self.illegal_action_penalty
            - self.unnecessary_action_penalty
        )


@dataclass(frozen=True)
class StepResult:
    next_state: MarketingState
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class TransitionConfig:
    do_nothing_opportunity_decay: float = 0.02
    do_nothing_fatigue_decay: float = 0.03
    defer_fatigue_decay: float = 0.07
    defer_conversion_decay: float = 0.02
    information_relevance_gain: float = 0.08
    information_fatigue_increase: float = 0.06
    information_intrusion_increase: float = 0.05
    reminder_boost: float = 0.14
    reminder_fatigue_increase: float = 0.09
    reminder_intrusion_increase: float = 0.08
    reminder_misuse_penalty: float = 0.12
    noise_scale: float = 0.03


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 42
    horizon: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 512
    train_epochs: int = 6
    minibatch_size: int = 64
    hidden_sizes: tuple[int, int] = (128, 128)
    deterministic_eval: bool = True


ArrayF32 = np.ndarray
ArrayBool = np.ndarray
