from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class OODDetector:
    model: IsolationForest
    feature_columns: list[str]

    @classmethod
    def create(cls, feature_columns: list[str], random_state: int = 42) -> "OODDetector":
        return cls(
            model=IsolationForest(n_estimators=120, contamination=0.05, random_state=random_state),
            feature_columns=feature_columns,
        )

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df[self.feature_columns])

    def score(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.model.decision_function(df[self.feature_columns])
        return (1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)).clip(0, 1)
