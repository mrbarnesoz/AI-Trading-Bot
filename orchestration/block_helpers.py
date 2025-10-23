"""Helpers for loading Prefect blocks with graceful fallbacks."""

from __future__ import annotations

from typing import Any, Dict, Optional


def load_json_block(name: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        from prefect.blocks.system import JSON
    except Exception:
        return default
    try:
        block = JSON.load(name)
        return block.value
    except Exception:
        return default


def load_secret_block(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        from prefect.blocks.system import Secret
    except Exception:
        return default
    try:
        secret = Secret.load(name)
        return secret.get()
    except Exception:
        return default
