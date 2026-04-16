from __future__ import annotations


def generate_message(action: str) -> str:
    templates = {
        "recommend_offer_a": "Based on your preferences, Offer A might be useful.",
        "recommend_offer_b": "You may benefit from Offer B this week.",
        "send_information": "Here is useful product information without promotion.",
        "send_reminder": "Friendly reminder: your saved items are still available.",
        "defer_action": "We will wait and learn before sending any campaign.",
        "do_nothing": "No message sent to avoid unnecessary contact.",
    }
    return templates.get(action, templates["do_nothing"])
