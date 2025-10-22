"""Evaluation helpers for model and strategy performance."""

from __future__ import annotations

import json
from typing import Dict


def pretty_print_metrics(metrics: Dict[str, float]) -> str:
    """Return a JSON-formatted string for metrics dictionaries."""
    return json.dumps(metrics, indent=2, sort_keys=True)
