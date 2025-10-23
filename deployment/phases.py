"""Deployment phase definitions and checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class DeploymentPhase:
    name: str
    duration_days: int
    mode: str
    objective: str
    promotion_check: Callable[[], bool]


class DeploymentPlan:
    def __init__(self, phases: List[DeploymentPhase]) -> None:
        self.phases = phases

    def next_phase(self, current_phase: str) -> DeploymentPhase | None:
        for idx, phase in enumerate(self.phases):
            if phase.name == current_phase and idx + 1 < len(self.phases):
                return self.phases[idx + 1]
        return None
