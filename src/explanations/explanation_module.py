from __future__ import annotations


def build_explanation(action: str, scores: dict[str, float], rule_reason: str | None = None) -> str:
    if action == "do_nothing":
        return f"No outreach selected to reduce intrusiveness ({rule_reason or 'safety policy'})."
    return (
        f"Action {action} selected with relevance={scores.get('offer_relevance_pred', 0):.2f}, "
        f"fatigue={scores.get('fatigue_risk_pred', 0):.2f}."
    )
