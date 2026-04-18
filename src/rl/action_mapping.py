from __future__ import annotations

ACTION_ID_TO_NAME = {
    0: "do_nothing",
    1: "defer_action",
    2: "send_information",
    3: "send_reminder",
}

ACTION_NAME_TO_ID = {name: idx for idx, name in ACTION_ID_TO_NAME.items()}

NUM_ACTIONS = 4
