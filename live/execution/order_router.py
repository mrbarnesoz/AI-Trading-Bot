"""Order routing and execution helpers."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


class OrderRouter:
    """Handle order submission for maker-first strategy."""

    def __init__(self, api_client) -> None:
        self.api_client = api_client

    async def fetch_market_state(self, symbol: str):
        return await self.api_client.get_market_state(symbol)

    async def execute(self, symbol: str, decision: Any) -> None:
        if is_dataclass(decision):
            payload: Dict[str, Any] = asdict(decision)
        elif isinstance(decision, dict):
            payload = decision
        else:
            raise TypeError("Decision must be a dataclass or mapping.")

        side = payload.get("side", "hold")
        size = float(payload.get("size", 0.0))
        if side in {"hold", "flat"} or size <= 0:
            return

        if payload.get("cross"):
            logger.debug("Crossing spread for %s", symbol)
        await self.api_client.submit_order(symbol, payload)
