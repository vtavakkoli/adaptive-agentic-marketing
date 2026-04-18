from __future__ import annotations

import torch
from torch.distributions import Categorical


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(mask, logits, torch.full_like(logits, neg_inf))


class MaskedCategorical:
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        self.masked_logits = apply_action_mask(logits, mask)
        self.dist = Categorical(logits=self.masked_logits)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    @property
    def probs(self) -> torch.Tensor:
        return self.dist.probs
