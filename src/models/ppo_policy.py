from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.rl.action_mapping import ACTION_ID_TO_NAME


class PPOPolicyModel:
    def __init__(self, model_path: str | Path, input_dim: int, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.input_dim = input_dim
        self.device = device
        from src.rl.ppo import CustomPPOAgent

        self.agent = CustomPPOAgent.load(str(self.model_path), input_dim=input_dim, device=device)

    def predict(self, state_vec: np.ndarray, action_mask: np.ndarray, deterministic: bool = True) -> dict[str, Any]:
        action, _, _, probs = self.agent.act(state_vec, action_mask, deterministic=deterministic)
        valid_probs = probs[action_mask]
        sorted_valid = np.sort(valid_probs)[::-1]
        top1 = float(sorted_valid[0]) if len(sorted_valid) > 0 else 0.0
        top2 = float(sorted_valid[1]) if len(sorted_valid) > 1 else 0.0
        entropy = float(-(probs[action_mask] * np.log(np.clip(probs[action_mask], 1e-8, 1.0))).sum())
        return {
            "action_id": action,
            "selected_action": ACTION_ID_TO_NAME[action],
            "confidence": top1,
            "policy_entropy": entropy,
            "action_margin": max(0.0, top1 - top2),
            "probs": probs.tolist(),
        }
