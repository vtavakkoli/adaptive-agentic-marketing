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
    "channel",
    "offer_id",
]


class XGBoostModule:
    def __init__(self) -> None:
        self._feature_columns: list[str] = []
        self.model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
        )

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        encoded = pd.get_dummies(df[FEATURES].copy(), columns=["channel", "offer_id"], dummy_na=False)
        if fit:
            self._feature_columns = list(encoded.columns)
            return encoded

        if not self._feature_columns:
            raise RuntimeError("Model feature space is not initialized. Call fit() before predict_scores().")

        aligned = encoded.reindex(columns=self._feature_columns, fill_value=0)
        return aligned

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(self._prepare_features(df, fit=True), df[TARGETS[0]])

    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        contact_risk = self.model.predict_proba(self._prepare_features(df, fit=False))[:, 1]
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
