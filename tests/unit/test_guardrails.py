from src.policy.guardrails import evaluate_guardrails


def test_guardrails_block_invalid_outreach() -> None:
    result = evaluate_guardrails(
        {
            "fatigue_score": 0.9,
            "intrusiveness_risk": 0.8,
            "frequency_7d": 6,
            "has_prior_offer_exposure": 0,
            "has_prior_engagement_on_offer": 0,
        },
        {"fatigue_threshold": 0.78, "intrusion_threshold": 0.72, "max_touches_7d": 5},
    )
    assert "send_information" not in result.allowed_actions
    assert "send_reminder" not in result.allowed_actions
