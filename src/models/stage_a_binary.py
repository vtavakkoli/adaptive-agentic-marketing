from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class StageABinaryModel:
    estimator: HistGradientBoostingClassifier
    feature_columns: list[str]

    @classmethod
    def create(cls, feature_columns: list[str], random_state: int = 42) -> "StageABinaryModel":
        est = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=180, random_state=random_state)
        return cls(estimator=est, feature_columns=feature_columns)

    def fit(self, df: pd.DataFrame, y_binary: pd.Series) -> None:
        self.estimator.fit(df[self.feature_columns], y_binary)

    def predict_proba(self, df: pd.DataFrame):
        return self.estimator.predict_proba(df[self.feature_columns])
