"""In-memory management of sub-positions per symbol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Literal, Optional


Side = Literal["long", "short"]


@dataclass
class SubPosition:
    """Discrete tranche of exposure for attribution and partial management."""

    id: str
    side: Side
    size: float
    entry_price: float
    stop_price: Optional[float] = None
    trail_price: Optional[float] = None
    strategy: str = "core"
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "trail_price": self.trail_price,
            "strategy": self.strategy,
            "opened_at": self.opened_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubPosition":
        opened_at = data.get("opened_at")
        if isinstance(opened_at, str):
            opened_at = datetime.fromisoformat(opened_at)
        if opened_at is None:
            opened_at = datetime.now(timezone.utc)
        return cls(
            id=data["id"],
            side=data["side"],
            size=float(data["size"]),
            entry_price=float(data["entry_price"]),
            stop_price=data.get("stop_price"),
            trail_price=data.get("trail_price"),
            strategy=data.get("strategy", "core"),
            opened_at=opened_at,
        )

    @property
    def signed_size(self) -> float:
        return self.size if self.side == "long" else -self.size


class PositionManager:
    """Tracks per-symbol sub-positions and desired net exposure."""

    def __init__(self) -> None:
        self._positions: Dict[str, Dict[str, SubPosition]] = {}
        self._targets: Dict[str, float] = {}

    def open(self, symbol: str, position: SubPosition) -> None:
        bucket = self._positions.setdefault(symbol, {})
        bucket[position.id] = position

    def close(self, symbol: str, sub_position_id: str) -> Optional[SubPosition]:
        bucket = self._positions.get(symbol)
        if not bucket:
            return None
        return bucket.pop(sub_position_id, None)

    def iter_symbol(self, symbol: str) -> Iterable[SubPosition]:
        return list(self._positions.get(symbol, {}).values())

    def net_position(self, symbol: str) -> float:
        return sum(pos.signed_size for pos in self.iter_symbol(symbol))

    def total_exposure(self) -> float:
        return sum(abs(pos.signed_size) for positions in self._positions.values() for pos in positions.values())

    def set_target(self, symbol: str, target: float) -> None:
        self._targets[symbol] = target

    def target(self, symbol: str) -> float:
        return self._targets.get(symbol, 0.0)

    def to_dict(self) -> dict:
        return {
            "positions": {
                symbol: {sub_id: position.to_dict() for sub_id, position in positions.items()}
                for symbol, positions in self._positions.items()
            },
            "targets": self._targets,
        }

    def load(self, payload: dict | None) -> None:
        if not payload:
            return
        positions = payload.get("positions", {})
        self._positions = {
            symbol: {sub_id: SubPosition.from_dict(data) for sub_id, data in bucket.items()}
            for symbol, bucket in positions.items()
        }
        self._targets = {symbol: float(value) for symbol, value in payload.get("targets", {}).items()}


__all__ = ["SubPosition", "PositionManager", "Side"]
