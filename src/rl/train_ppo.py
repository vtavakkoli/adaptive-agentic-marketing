from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.config import load_yaml
from src.rl.environment import MarketingMDP
from src.rl.ppo import CustomPPOAgent
from src.rl.reward import RewardWeights
from src.rl.types import PPOConfig, TransitionConfig
from src.utils.logging_utils import configure_logging, log_event


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom PPO marketing agent")
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--model-path", default="outputs/models/adaptive_ppo_agent.pt")
    parser.add_argument("--timesteps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--config", default="configs/adaptive_hierarchical.yaml")
    args = parser.parse_args()

    logger = configure_logging()
    cfg = load_yaml(args.config)
    ppo_cfg = cfg.get("ppo", {})

    train_df = pd.read_csv(args.train_path)
    transition_cfg = TransitionConfig(**ppo_cfg.get("transition", {}))
    reward_weights = RewardWeights(**ppo_cfg.get("reward_weights", {}))
    ppo_conf = PPOConfig(
        **{k: v for k, v in ppo_cfg.items() if k in PPOConfig.__dataclass_fields__},
        seed=args.seed,
        horizon=args.horizon,
    )

    env = MarketingMDP(
        dataset=train_df,
        horizon=ppo_conf.horizon,
        transition_cfg=transition_cfg,
        reward_weights=reward_weights,
        guardrail_cfg=cfg.get("guardrails", {}),
        seed=args.seed,
    )
    input_dim = len(env.encode_state(env.reset(seed=args.seed)))
    agent = CustomPPOAgent(input_dim=input_dim, cfg=ppo_conf)
    metrics = agent.train(env=env, timesteps=args.timesteps)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "train_rows": len(train_df),
        "timesteps": args.timesteps,
    }
    agent.save(str(model_path), metadata=metadata)

    summary = {
        "model_path": str(model_path),
        "metrics": metrics,
        "ppo_config": asdict(ppo_conf),
        "transition_config": asdict(transition_cfg),
        "reward_weights": asdict(reward_weights),
        "metadata": metadata,
    }
    summary_path = model_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_event(logger, "ppo_training_complete", model_path=str(model_path), **metrics)


if __name__ == "__main__":
    main()
