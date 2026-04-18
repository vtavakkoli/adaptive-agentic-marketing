import pytest

pytest.importorskip("torch")

from pathlib import Path

import numpy as np
import pandas as pd

from src.agentic.controller import AdaptiveAgenticController
from src.rl.action_mapping import ACTION_ID_TO_NAME, ACTION_NAME_TO_ID
from src.rl.distributions import MaskedCategorical
from src.rl.environment import MarketingMDP
from src.rl.ppo import CustomPPOAgent
from src.rl.reward import RewardWeights, compute_reward
from src.rl.state import state_from_row
from src.rl.transition import apply_transition
from src.rl.types import PPOConfig, TransitionConfig


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "recency_days": 7,
                "frequency_7d": 1,
                "avg_basket_value": 30.0,
                "campaign_touches_30d": 2,
                "prior_response_rate": 0.5,
                "need_score": 0.7,
                "fatigue_score": 0.2,
                "intrusiveness_risk": 0.2,
                "offer_relevance": 0.8,
                "channel": "email",
                "offer_id": "offer_x",
            }
        ]
    )


def test_action_mapping_stable() -> None:
    assert ACTION_ID_TO_NAME[0] == "do_nothing"
    assert ACTION_NAME_TO_ID["send_reminder"] == 3


def test_valid_action_mask_includes_do_nothing() -> None:
    env = MarketingMDP(_dataset(), guardrail_cfg={"fatigue_threshold": 0.1, "intrusion_threshold": 0.1, "max_touches_7d": 0})
    env.reset(seed=42)
    mask = env.valid_action_mask()
    assert bool(mask[0]) is True


def test_transition_clipping() -> None:
    state = state_from_row(_dataset().iloc[0].to_dict())
    next_state = apply_transition(
        state,
        action=3,
        cfg=TransitionConfig(reminder_fatigue_increase=10.0, reminder_intrusion_increase=10.0),
        rng=np.random.default_rng(1),
        reminder_valid=False,
    )
    assert 0.0 <= next_state.fatigue_score <= 1.0
    assert 0.0 <= next_state.intrusiveness_risk <= 1.0


def test_reward_decomposition_total() -> None:
    state = state_from_row(_dataset().iloc[0].to_dict())
    reward = compute_reward(state, action=2, mask_valid=True, weights=RewardWeights())
    assert isinstance(reward.total, float)


def test_masked_sampling_never_returns_invalid() -> None:
    import torch

    logits = torch.tensor([[0.2, 0.1, 0.3, 3.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False, False, False]])
    dist = MaskedCategorical(logits, mask)
    samples = [int(dist.sample().item()) for _ in range(20)]
    assert set(samples) == {0}


def test_ppo_loss_runs_without_shape_errors() -> None:
    env = MarketingMDP(_dataset(), seed=7)
    env.reset(seed=7)
    cfg = PPOConfig(rollout_steps=16, train_epochs=1, minibatch_size=8, hidden_sizes=(32, 32))
    agent = CustomPPOAgent(input_dim=16, cfg=cfg)
    rollout = agent.collect_rollout(env)
    metrics = agent.update(rollout)
    assert "policy_loss" in metrics


def test_model_save_load_roundtrip(tmp_path: Path) -> None:
    cfg = PPOConfig(rollout_steps=8, train_epochs=1, minibatch_size=4, hidden_sizes=(16, 16))
    agent = CustomPPOAgent(input_dim=16, cfg=cfg)
    model_path = tmp_path / "ppo.pt"
    agent.save(str(model_path))
    loaded = CustomPPOAgent.load(str(model_path), input_dim=16)
    assert type(loaded).__name__ == "CustomPPOAgent"


def test_controller_integration_adaptive_ppo_agent(tmp_path: Path) -> None:
    cfg = PPOConfig(rollout_steps=8, train_epochs=1, minibatch_size=4, hidden_sizes=(16, 16))
    agent = CustomPPOAgent(input_dim=16, cfg=cfg)
    model_path = tmp_path / "adaptive_ppo_agent.pt"
    agent.save(str(model_path))

    controller = AdaptiveAgenticController(
        {
            "model_path": "outputs/models/missing_xgb.joblib",
            "rules": {},
            "guardrails": {},
            "slm": {"enabled": False},
            "ppo": {"model_path": str(model_path), "deterministic_eval": True},
        }
    )
    out = controller.decide(_dataset().iloc[0].to_dict(), mode="adaptive_ppo_agent")
    assert out["selected_action"] in ACTION_NAME_TO_ID
    assert "policy_entropy" in out
