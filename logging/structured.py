"""Structured logging helpers for trading decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class StructuredLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: Dict) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event) + "\n")
