from __future__ import annotations

import numpy as np
import pandas as pd


ACTION_ORDER = [
    "do_nothing",
    "defer_action",
    "send_information",
    "send_reminder",
]


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["need_score"] = (1 / (1 + np.exp((out["recency_days"] - 30) / 12))).clip(0, 1)
    out["fatigue_score"] = (out["campaign_touches_30d"] / 12).clip(0, 1)
    out["intrusiveness_risk"] = (
        0.5 * out["fatigue_score"] + 0.5 * (out["frequency_7d"] / 7).clip(0, 1)
    ).clip(0, 1)
    out["offer_relevance"] = (
        0.6 * out["prior_response_rate"] + 0.4 * out["need_score"]
    ).clip(0, 1)
    out["no_action_preferred"] = (
        (out["fatigue_score"] > 0.66)
        | (out["intrusiveness_risk"] > 0.70)
        | (out["offer_relevance"] < 0.35)
    ).astype(int)

    out["action_class"] = np.select(
        [
            out["no_action_preferred"] == 1,
            out["need_score"] > 0.70,
            (out["need_score"] > 0.45) & (out["offer_relevance"] > 0.45),
        ],
        ["do_nothing", "send_reminder", "send_information"],
        default="defer_action",
    )
    return out
