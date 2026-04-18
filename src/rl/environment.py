from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_builder import build_features
from src.policy.guardrails import evaluate_guardrails
from src.rl.action_mapping import ACTION_NAME_TO_ID
from src.rl.reward import RewardWeights, compute_reward
from src.rl.state import state_from_row
from src.rl.state_encoder import encode_state
from src.rl.transition import apply_transition
from src.rl.types import MarketingState, StepResult, TransitionConfig


class MarketingMDP:
    """Custom simulator-backed MDP for sequential marketing decisions.

    This environment intentionally does not depend on Gymnasium. It bootstraps
    initial states from static tabular rows and applies configurable stochastic
    transition and reward models to synthesize trajectories for policy learning.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        horizon: int = 8,
        transition_cfg: TransitionConfig | None = None,
        reward_weights: RewardWeights | None = None,
        guardrail_cfg: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = build_features(dataset.copy())
        self.horizon = horizon
        self.transition_cfg = transition_cfg or TransitionConfig()
        self.reward_weights = reward_weights or RewardWeights()
        self.guardrail_cfg = guardrail_cfg or {}
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.state: MarketingState | None = None
        self.step_idx = 0

    def sample_initial_state(self) -> MarketingState:
        idx = int(self.rng.integers(0, len(self.dataset)))
        row = self.dataset.iloc[idx].to_dict()
        return state_from_row(row)

    def reset(self, seed: int | None = None) -> MarketingState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_idx = 0
        self.state = self.sample_initial_state()
        return self.state

    def is_terminal(self, state: MarketingState) -> bool:
        return state.readiness_score < 0.05 or state.recency_days >= 365.0

    def valid_action_mask(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment state is not initialized")
        guard = evaluate_guardrails(asdict(self.state), self.guardrail_cfg)
        mask = np.zeros(4, dtype=bool)
        for action_name in guard.allowed_actions:
            mask[ACTION_NAME_TO_ID[action_name]] = True
        mask[ACTION_NAME_TO_ID["do_nothing"]] = True
        return mask

    def encode_state(self, state: MarketingState | None = None) -> np.ndarray:
        working_state = state or self.state
        if working_state is None:
            raise RuntimeError("Environment state is not initialized")
        return encode_state(working_state)

    def step(self, action: int) -> StepResult:
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        if action not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid action id: {action}")

        mask = self.valid_action_mask()
        originally_valid = bool(mask[action])
        safe_action = action if originally_valid else ACTION_NAME_TO_ID["do_nothing"]

        reminder_valid = bool(mask[ACTION_NAME_TO_ID["send_reminder"]])
        reward_breakdown = compute_reward(self.state, action=safe_action, mask_valid=originally_valid, weights=self.reward_weights)
        next_state = apply_transition(self.state, safe_action, self.transition_cfg, self.rng, reminder_valid=reminder_valid)

        self.state = next_state
        self.step_idx += 1

        terminated = self.is_terminal(next_state)
        truncated = self.step_idx >= self.horizon
        info = {
            "applied_action": safe_action,
            "invalid_action_fallback": not originally_valid,
            "reward_breakdown": asdict(reward_breakdown),
        }
        return StepResult(
            next_state=next_state,
            reward=float(reward_breakdown.total),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
