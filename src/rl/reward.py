from __future__ import annotations

from dataclasses import dataclass

from src.rl.action_mapping import ACTION_ID_TO_NAME
from src.rl.types import MarketingState, RewardBreakdown


@dataclass(frozen=True)
class RewardWeights:
    conversion: float = 2.0
    engagement: float = 0.8
    relevance: float = 0.4
    fatigue: float = 0.9
    intrusiveness: float = 0.8
    over_contact: float = 0.7
    illegal_action: float = 3.0
    unnecessary_action: float = 1.0
    abstention: float = 0.3


def compute_reward(state: MarketingState, action: int, mask_valid: bool, weights: RewardWeights) -> RewardBreakdown:
    name = ACTION_ID_TO_NAME[action]
    outreach = name in {"send_information", "send_reminder"}
    contact_pressure = state.frequency_7d / 7.0

    conversion_reward = weights.conversion * (state.readiness_score * state.offer_relevance) * (1.0 if outreach else 0.2)
    engagement_reward = weights.engagement * state.prior_response_rate * (1.0 if outreach else 0.3)
    relevance_reward = weights.relevance * state.offer_relevance * (1.0 if outreach else 0.0)

    fatigue_penalty = weights.fatigue * state.fatigue_score * (1.2 if outreach else 0.4)
    intrusiveness_penalty = weights.intrusiveness * state.intrusiveness_risk * (1.2 if outreach else 0.3)
    over_contact_penalty = weights.over_contact * max(0.0, contact_pressure - 0.55) * (1.5 if outreach else 0.2)

    illegal_action_penalty = weights.illegal_action if not mask_valid else 0.0
    unnecessary_action_penalty = 0.0
    if outreach and state.offer_relevance < 0.35 and state.readiness_score < 0.35:
        unnecessary_action_penalty = weights.unnecessary_action

    abstention_reward = 0.0
    if name == "do_nothing" and (state.fatigue_score > 0.75 or state.intrusiveness_risk > 0.7):
        abstention_reward = weights.abstention

    return RewardBreakdown(
        conversion_reward=conversion_reward,
        engagement_reward=engagement_reward,
        relevance_reward=relevance_reward,
        fatigue_penalty=fatigue_penalty,
        intrusiveness_penalty=intrusiveness_penalty,
        over_contact_penalty=over_contact_penalty,
        illegal_action_penalty=illegal_action_penalty,
        unnecessary_action_penalty=unnecessary_action_penalty,
        abstention_reward=abstention_reward,
    )
