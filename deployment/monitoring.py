"""Monitoring and alerting scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class AlertRule:
    name: str
    condition: Callable[..., bool]
    action: Callable[..., None]


class MonitoringSuite:
    def __init__(self) -> None:
        self.alerts: Dict[str, AlertRule] = {}

    def register_alert(self, rule: AlertRule) -> None:
        self.alerts[rule.name] = rule

    def evaluate(self, metrics: Dict[str, float]) -> None:
        for rule in self.alerts.values():
            if rule.condition(metrics):
                rule.action(metrics)
