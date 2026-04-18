from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_yaml
from src.data.feature_builder import build_features
from src.models.ppo_policy import PPOPolicyModel
from src.policy.guardrails import evaluate_guardrails
from src.rl.action_mapping import ACTION_NAME_TO_ID
from src.rl.state import state_from_row
from src.rl.state_encoder import encode_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate custom PPO marketing agent")
    parser.add_argument("--eval-path", default="data/processed/test.csv")
    parser.add_argument("--model-path", default="outputs/models/adaptive_ppo_agent.pt")
    parser.add_argument("--output-path", default="outputs/predictions/adaptive_ppo_agent/eval/predictions.csv")
    parser.add_argument("--config", default="configs/adaptive_hierarchical.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    df = build_features(pd.read_csv(args.eval_path))
    model = PPOPolicyModel(model_path=args.model_path, input_dim=16)

    rows: list[dict] = []
    deterministic = bool(cfg.get("ppo", {}).get("deterministic_eval", True))
    for row in df.to_dict(orient="records"):
        state = state_from_row(row)
        mask = [False] * 4
        guard = evaluate_guardrails(row, cfg.get("guardrails", {}))
        for action_name in guard.allowed_actions:
            mask[ACTION_NAME_TO_ID[action_name]] = True
        mask[ACTION_NAME_TO_ID["do_nothing"]] = True
        out = model.predict(encode_state(state), action_mask=pd.Series(mask, dtype=bool).to_numpy(), deterministic=deterministic)
        rows.append(
            {
                "selected_action": out["selected_action"],
                "confidence": out["confidence"],
                "no_action": out["selected_action"] == "do_nothing",
                "supporting_scores": {"rl_probs": out["probs"]},
                "policy_entropy": out["policy_entropy"],
                "action_margin": out["action_margin"],
            }
        )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
