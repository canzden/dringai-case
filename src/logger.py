from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


class VoiceAssistantLogger:
    CURRENT_PATH = Path(__file__).resolve().parent
    """Append one JSON object per line to a new created conversation log file"""

    def __init__(self):
        self.log_dir = os.path.join(self.CURRENT_PATH, os.getenv("LOG_DIR") or "../data/logs")
        self._lock = threading.Lock()
        self.log_filename = datetime.now(timezone.utc).isoformat(timespec="seconds")
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def log_file_path(self) -> str:
        return os.path.join(self.log_dir, f"{self.log_filename}.jsonl")

    def log(self, turn_id: int, user_text: str, assistant_text: str) -> None:
        rec: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "turn_id": turn_id,
            "user_text": user_text,
            "assistant_text": assistant_text,
        }
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
