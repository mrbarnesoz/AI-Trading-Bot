"""Retraining scheduler stubs per regime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict


@dataclass
class RetrainSchedule:
    cadence_days: int
    shadow_days: int


class RetrainScheduler:
    def __init__(self, schedules: Dict[str, RetrainSchedule]) -> None:
        self.schedules = schedules
        self.last_run: Dict[str, datetime] = {}

    def due(self, regime: str, now: datetime) -> bool:
        last = self.last_run.get(regime)
        if last is None:
            return True
        return now - last >= timedelta(days=self.schedules[regime].cadence_days)

    def mark_run(self, regime: str, now: datetime) -> None:
        self.last_run[regime] = now
