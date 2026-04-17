from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def top_confusion_pairs(df: pd.DataFrame, preds: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    y_true = df["action_class"].astype(str).tolist()
    y_pred = [str(p.get("selected_action", "unknown")) for p in preds]
    pairs = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    return [{"true": t, "pred": p, "count": c} for (t, p), c in pairs.most_common(top_k)]
