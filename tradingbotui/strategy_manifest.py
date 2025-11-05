"""Helpers for loading strategy metadata such as recommended timeframes."""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import yaml

MANIFEST_PATH = Path("configs") / "strategy_manifest.yaml"

logger = logging.getLogger("tradingbot.ui.strategy_manifest")


class StrategyManifestEntry(TypedDict, total=False):
    label: str
    description: str
    timeframes: List[str]
    default: str


Manifest = Dict[str, StrategyManifestEntry]


def _load_manifest() -> Manifest:
    if not MANIFEST_PATH.exists():
        logger.warning("Strategy manifest not found at %s", MANIFEST_PATH)
        return {}
    try:
        raw = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to read strategy manifest: %s", exc)
        return {}
    manifest: Manifest = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            timeframes = value.get("timeframes") or []
            if not isinstance(timeframes, list):
                timeframes = []
            entry: StrategyManifestEntry = {
                "label": str(value.get("label", key.replace("_", " ").title())),
                "description": value.get("description") or "",
                "timeframes": [str(item) for item in timeframes],
            }
            default_tf = value.get("default")
            if isinstance(default_tf, str):
                entry["default"] = default_tf
            manifest[str(key)] = entry
    return manifest


@functools.lru_cache(maxsize=1)
def get_strategy_manifest() -> Manifest:
    """Return the cached strategy manifest."""
    return _load_manifest()


def get_timeframes_for_strategy(strategy: str) -> StrategyManifestEntry:
    """Return metadata for a strategy, falling back to an empty entry."""
    manifest = get_strategy_manifest()
    return manifest.get(strategy, {"label": strategy, "description": "", "timeframes": []})
