import pandas as pd

from src.data.label_audit import run_label_audit


def test_label_audit_detects_missing_reminder_prereq() -> None:
    df = pd.DataFrame(
        [
            {"action_class": "send_reminder", "has_prior_offer_exposure": 0, "has_prior_engagement_on_offer": 0, "has_incomplete_intent": 0, "abandoned_action_flag": 0},
            {"action_class": "send_information", "has_prior_offer_exposure": 1, "has_prior_engagement_on_offer": 0, "has_incomplete_intent": 0, "abandoned_action_flag": 0},
        ]
    )
    out = run_label_audit(df)
    assert out["send_reminder_missing_prerequisites"] == 1
