from src.policy.rules_engine import apply_rules


def test_high_fatigue_forces_no_action() -> None:
    out = apply_rules(
        {"frequency_7d": 1, "fatigue_score": 0.8, "offer_relevance": 0.8, "offer_id": "offer_a"},
        {"frequency_cap_7d": 5, "fatigue_do_nothing_threshold": 0.7, "min_relevance_threshold": 0.35},
    )
    assert out.action == "do_nothing"
