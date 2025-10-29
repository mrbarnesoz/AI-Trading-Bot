"""Order routing and execution helpers."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from live.risk.trailing import TrailingManager

logger = logging.getLogger(__name__)


class OrderRouter:
    """Handle order submission for maker-first strategy."""

    def __init__(self, api_client, trailing_manager: Optional[TrailingManager] = None) -> None:
        self.api_client = api_client
        self.trailing_manager = trailing_manager

    async def fetch_market_state(self, symbol: str):
        return await self.api_client.get_market_state(symbol)

    async def execute(self, symbol: str, decision: Any, market_state=None) -> None:
        if is_dataclass(decision):
            payload: Dict[str, Any] = asdict(decision)
        elif isinstance(decision, dict):
            payload = decision
        else:
            raise TypeError("Decision must be a dataclass or mapping.")

        side = payload.get("side", "hold")
        size = float(payload.get("size", 0.0))
        if side in {"hold", "flat"} or size <= 0:
            if self.trailing_manager:
                self.trailing_manager.record_execution(symbol, payload, market_state)
            return

        if payload.get("cross"):
            logger.debug("Crossing spread for %s", symbol)
        await self.api_client.submit_order(symbol, payload)
        if self.trailing_manager:
            self.trailing_manager.record_execution(symbol, payload, market_state)
