from __future__ import annotations

import torch
from torch import nn


class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], num_actions: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, num_actions)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value
