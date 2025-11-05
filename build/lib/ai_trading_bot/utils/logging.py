"""Logging helpers for the AI Trading Bot."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logging(log_level: int = logging.INFO, log_file: Optional[Path | str] = Path("logs") / "app.log") -> None:
    """Configure root logger with console and optional file handlers."""
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Avoid duplicate handlers if configure_logging is called multiple times.
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(log_level)
        return

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(file_handler)
