import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "audit_trail.jsonl")


def log_step(step_name: str, details: dict):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step_name,
        "details": details,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_trail(limit: int = 50):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()[-limit:]
    return [json.loads(line) for line in lines]
