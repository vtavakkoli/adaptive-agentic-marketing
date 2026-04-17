from __future__ import annotations


def generate_message(action: str) -> str:
    templates = {
        "send_information": "Here is useful product information without promotion.",
        "send_reminder": "Friendly reminder: your saved items are still available.",
        "defer_action": "We will wait and learn before sending any campaign.",
        "do_nothing": "No message sent to avoid unnecessary contact.",
    }
    return templates.get(action, templates["do_nothing"])
