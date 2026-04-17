from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

TARGETS = [
    "no_action_preferred",
]
FEATURES = [
    "recency_days",
    "frequency_7d",
    "avg_basket_value",
    "campaign_touches_30d",
    "prior_response_rate",
    "need_score",
    "fatigue_score",
    "intrusiveness_risk",
    "offer_relevance",
]


class XGBoostModule:
    def __init__(self) -> None:
        self.model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
        )

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df[FEATURES], df[TARGETS[0]])

    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        contact_risk = self.model.predict_proba(df[FEATURES])[:, 1]
        response_propensity = (
            0.6 * df["prior_response_rate"].astype(float) + 0.4 * (1.0 - contact_risk)
        ).clip(0, 1)
        return pd.DataFrame(
            {
                "contact_risk_pred": contact_risk,
                "response_propensity_pred": response_propensity,
            }
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, p)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
