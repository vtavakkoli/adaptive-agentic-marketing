from src.policy.controller import HierarchicalPolicyController


def test_controller_uncertainty_falls_back_to_defer() -> None:
    controller = HierarchicalPolicyController(
        {
            "controller": {
                "binary_action_threshold": 0.4,
                "min_action_confidence": 0.7,
                "top2_gap_threshold": 0.2,
                "risk_abstain_threshold": 0.95,
            },
            "guardrails": {},
        }
    )
    decision = controller.decide(
        {"intrusiveness_risk": 0.1, "has_prior_offer_exposure": 1, "has_prior_engagement_on_offer": 1},
        stage_a_probs={"do_nothing": 0.2, "action": 0.8},
        stage_b_probs={"defer_action": 0.34, "send_information": 0.35, "send_reminder": 0.31},
    )
    assert decision.final_action == "defer_action"
    assert decision.fallback_reason == "uncertainty_fallback"
