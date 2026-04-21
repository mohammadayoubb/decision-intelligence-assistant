import json
from datetime import datetime
from pathlib import Path

from app.config import LOGS_DIR


LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "query_logs.jsonl"


def log_query_result(payload: dict):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        **payload
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")