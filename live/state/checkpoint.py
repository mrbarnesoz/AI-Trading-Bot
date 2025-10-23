"""State checkpoint persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from live.risk.guardrails import RiskGuardrails
from live.state.positions import PositionManager


class StateCheckpoint:
    def __init__(
        self,
        path: Path,
        risk: Optional[RiskGuardrails] = None,
        position_manager: Optional[PositionManager] = None,
    ) -> None:
        self.path = path
        self.risk = risk
        self.position_manager = position_manager or (risk.positions if risk else None)

    def bind(self, risk: RiskGuardrails) -> None:
        self.risk = risk
        if self.position_manager is None:
            self.position_manager = risk.positions

    async def persist(self) -> None:
        if self.risk is None or self.position_manager is None:
            raise RuntimeError("StateCheckpoint requires risk and position manager to persist state.")
        payload = {
            "positions": self.position_manager.to_dict(),
            "risk": self.risk.snapshot(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.risk.mark_checkpoint()

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        positions = data.get("positions", {})
        risk_state = data.get("risk", {})
        if self.position_manager:
            self.position_manager.load(positions)
        if self.risk:
            self.risk.restore(risk_state)
