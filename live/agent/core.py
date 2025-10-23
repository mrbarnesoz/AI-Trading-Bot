"""Embedded trading agent core orchestration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from live.execution.order_router import OrderRouter
from live.policy.decision import DecisionPolicy
from live.risk.guardrails import RiskGuardrails
from live.state.checkpoint import StateCheckpoint

logger = logging.getLogger(__name__)


@dataclass
class SymbolConfig:
    symbol: str
    regime: str


class EmbeddedAgent:
    """Single-process, per-symbol async trading agent."""

    def __init__(
        self,
        symbols: Dict[str, SymbolConfig],
        policy: DecisionPolicy,
        risk: RiskGuardrails,
        router: OrderRouter,
        checkpoint: StateCheckpoint,
    ) -> None:
        self.symbols = symbols
        self.policy = policy
        self.risk = risk
        self.router = router
        self.checkpoint = checkpoint
        self._tasks: Dict[str, asyncio.Task] = {}
        self.checkpoint.bind(self.risk)
        self.checkpoint.load()

    async def start(self) -> None:
        for symbol, cfg in self.symbols.items():
            task = asyncio.create_task(self._run_symbol_loop(symbol, cfg))
            self._tasks[symbol] = task

    async def _run_symbol_loop(self, symbol: str, cfg: SymbolConfig) -> None:
        logger.info("Starting loop for %s", symbol)
        while True:
            market_state = await self.router.fetch_market_state(symbol)
            decision = self.policy.decide(symbol, market_state)
            if self.risk.is_allowed(symbol, decision, market_state):
                await self.router.execute(symbol, decision)
            if self.risk.should_checkpoint():
                await self.checkpoint.persist()
            await asyncio.sleep(DecisionPolicy.sampling_interval(cfg.regime))
