from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class StageBActionModel:
    estimator: HistGradientBoostingClassifier
    feature_columns: list[str]
    classes: list[str]

    @classmethod
    def create(cls, feature_columns: list[str], random_state: int = 42) -> "StageBActionModel":
        est = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=220, random_state=random_state)
        return cls(
            estimator=est,
            feature_columns=feature_columns,
            classes=["defer_action", "send_information", "send_reminder"],
        )

    def fit(self, df: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> None:
        self.estimator.fit(df[self.feature_columns], y, sample_weight=sample_weight)

    def predict_proba(self, df: pd.DataFrame):
        return self.estimator.predict_proba(df[self.feature_columns])
