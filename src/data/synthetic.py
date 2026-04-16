from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    n_customers: int = 500
    n_rows: int = 4000
    seed: int = 42


def generate_synthetic_dataset(cfg: SyntheticConfig, out_dir: str | Path) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    customer_ids = [f"C{idx:05d}" for idx in range(cfg.n_customers)]
    df = pd.DataFrame(
        {
            "customer_id": rng.choice(customer_ids, cfg.n_rows),
            "recency_days": rng.integers(0, 120, cfg.n_rows),
            "frequency_7d": rng.integers(0, 8, cfg.n_rows),
            "avg_basket_value": rng.normal(45, 18, cfg.n_rows).clip(2, 250),
            "offer_id": rng.choice(["offer_a", "offer_b", "info", "reminder"], cfg.n_rows),
            "channel": rng.choice(["email", "sms", "app"], cfg.n_rows),
            "campaign_touches_30d": rng.integers(0, 14, cfg.n_rows),
            "prior_response_rate": rng.random(cfg.n_rows),
        }
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "synthetic_customer_events.csv"
    df.to_csv(raw_path, index=False)
    return df
