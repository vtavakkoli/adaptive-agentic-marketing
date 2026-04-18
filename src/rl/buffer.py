from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutBatch:
    states: np.ndarray
    actions: np.ndarray
    masks: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    next_value: float


class RolloutBuffer:
    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.masks: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        mask: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.masks.append(mask)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.log_probs.append(log_prob)
        self.values.append(value)

    def as_batch(self, next_value: float) -> RolloutBatch:
        return RolloutBatch(
            states=np.asarray(self.states, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.int64),
            masks=np.asarray(self.masks, dtype=bool),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            dones=np.asarray(self.dones, dtype=np.float32),
            log_probs=np.asarray(self.log_probs, dtype=np.float32),
            values=np.asarray(self.values, dtype=np.float32),
            next_value=float(next_value),
        )
