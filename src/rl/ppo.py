from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import nn

from src.rl.action_mapping import NUM_ACTIONS
from src.rl.buffer import RolloutBuffer
from src.rl.distributions import MaskedCategorical
from src.rl.gae import compute_gae
from src.rl.networks import ActorCriticPolicy
from src.rl.types import PPOConfig


class CustomPPOAgent:
    def __init__(self, input_dim: int, cfg: PPOConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.policy = ActorCriticPolicy(input_dim=input_dim, hidden_sizes=cfg.hidden_sizes, num_actions=NUM_ACTIONS).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self._set_seed(cfg.seed)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def act(self, state: np.ndarray, mask: np.ndarray, deterministic: bool = False) -> tuple[int, float, float, np.ndarray]:
        state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        mask_t = torch.from_numpy(mask).bool().to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.policy(state_t)
            dist = MaskedCategorical(logits, mask_t)
            action_t = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
            log_prob_t = dist.log_prob(action_t)
            probs = dist.probs.squeeze(0).cpu().numpy()
        return int(action_t.item()), float(log_prob_t.item()), float(value.item()), probs

    def collect_rollout(self, env: Any) -> RolloutBuffer:
        buf = RolloutBuffer()
        state = env.reset(seed=self.cfg.seed)
        for _ in range(self.cfg.rollout_steps):
            encoded = env.encode_state(state)
            mask = env.valid_action_mask()
            action, log_prob, value, _ = self.act(encoded, mask, deterministic=False)
            step = env.step(action)
            done = step.terminated or step.truncated
            buf.add(encoded, action, mask, step.reward, done, log_prob, value)
            state = env.reset() if done else step.next_state
        next_value = 0.0
        if env.state is not None:
            with torch.no_grad():
                s = torch.from_numpy(env.encode_state()).float().to(self.device).unsqueeze(0)
                _, v = self.policy(s)
                next_value = float(v.item())
        buf.next_value = next_value  # type: ignore[attr-defined]
        return buf

    def update(self, rollout: RolloutBuffer) -> dict[str, float]:
        batch = rollout.as_batch(next_value=getattr(rollout, "next_value", 0.0))
        adv, ret = compute_gae(batch.rewards, batch.values, batch.dones, batch.next_value, self.cfg.gamma, self.cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states = torch.from_numpy(batch.states).float().to(self.device)
        actions = torch.from_numpy(batch.actions).long().to(self.device)
        masks = torch.from_numpy(batch.masks).bool().to(self.device)
        old_log_probs = torch.from_numpy(batch.log_probs).float().to(self.device)
        returns = torch.from_numpy(ret).float().to(self.device)
        advantages = torch.from_numpy(adv).float().to(self.device)

        n = len(batch.actions)
        idxs = np.arange(n)
        last_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.cfg.train_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.cfg.minibatch_size):
                mb_idx = idxs[start : start + self.cfg.minibatch_size]
                s_mb = states[mb_idx]
                a_mb = actions[mb_idx]
                m_mb = masks[mb_idx]
                old_lp_mb = old_log_probs[mb_idx]
                ret_mb = returns[mb_idx]
                adv_mb = advantages[mb_idx]

                logits, values = self.policy(s_mb)
                dist = MaskedCategorical(logits, m_mb)
                new_log_probs = dist.log_prob(a_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_lp_mb)
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                policy_loss = -torch.min(ratio * adv_mb, clipped * adv_mb).mean()
                value_loss = nn.functional.mse_loss(values, ret_mb)

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                last_metrics = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }
        return last_metrics

    def train(self, env: Any, timesteps: int) -> dict[str, float]:
        updates = max(1, timesteps // self.cfg.rollout_steps)
        metrics: dict[str, float] = {}
        for _ in range(updates):
            rollout = self.collect_rollout(env)
            metrics = self.update(rollout)
        return metrics

    def save(self, model_path: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "state_dict": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "ppo_config": asdict(self.cfg),
            "metadata": metadata or {},
        }
        torch.save(payload, model_path)

    @classmethod
    def load(cls, model_path: str, input_dim: int, device: str = "cpu") -> "CustomPPOAgent":
        payload = torch.load(model_path, map_location=device)
        cfg = PPOConfig(**payload["ppo_config"])
        agent = cls(input_dim=input_dim, cfg=cfg, device=device)
        agent.policy.load_state_dict(payload["state_dict"])
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state:
            agent.optimizer.load_state_dict(optimizer_state)
        return agent
