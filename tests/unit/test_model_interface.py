import pandas as pd

from src.models.xgboost_module import FEATURES, XGBoostModule


def test_model_fit_predict_interface() -> None:
    df = pd.DataFrame(
        {
            "recency_days": [1, 20, 50, 5],
            "frequency_7d": [1, 5, 3, 0],
            "avg_basket_value": [10.0, 40.0, 15.0, 90.0],
            "campaign_touches_30d": [1, 8, 4, 0],
            "prior_response_rate": [0.8, 0.2, 0.5, 0.9],
            "need_score": [0.9, 0.4, 0.5, 0.7],
            "fatigue_score": [0.1, 0.8, 0.4, 0.1],
            "intrusiveness_risk": [0.1, 0.9, 0.5, 0.2],
            "offer_relevance": [0.9, 0.2, 0.6, 0.8],
            "no_action_preferred": [0, 1, 0, 0],
        }
    )
    model = XGBoostModule()
    model.fit(df)
    out = model.predict_scores(df)
    assert not out.empty
    assert set(FEATURES).issubset(df.columns)
