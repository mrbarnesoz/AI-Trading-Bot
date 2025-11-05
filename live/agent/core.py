"""Embedded trading agent core orchestration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from live.execution.order_router import OrderRouter
from live.policy.decision import DecisionPolicy
from live.risk.guardrails import RiskGuardrails
from live.risk.trailing import TrailingManager
from live.state.checkpoint import StateCheckpoint
from ai_trading_bot.meta.select import MetaStrategySelector

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
        trailing: TrailingManager | None = None,
        meta_selector: MetaStrategySelector | None = None,
    ) -> None:
        self.symbols = symbols
        self.policy = policy
        self.risk = risk
        self.router = router
        self.checkpoint = checkpoint
        self.trailing = trailing
        self.meta_selector = meta_selector
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
            if self.trailing and self.trailing.enabled():
                trail_events = self.trailing.update(symbol, market_state)
                for event in trail_events:
                    if self.risk.is_allowed(symbol, event, market_state):
                        await self.router.execute(symbol, event, market_state)
            decision = self.policy.decide(symbol, market_state)
            if self.risk.is_allowed(symbol, decision, market_state):
                await self.router.execute(symbol, decision, market_state)
            if self.risk.should_checkpoint():
                await self.checkpoint.persist()
                if self.trailing:
                    self.trailing.flush()
                if self.meta_selector:
                    self.meta_selector.flush()
                self.risk.mark_checkpoint()
            await asyncio.sleep(DecisionPolicy.sampling_interval(cfg.regime))
