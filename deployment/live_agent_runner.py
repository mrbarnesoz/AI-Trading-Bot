"""Utility to bootstrap the embedded live agent with configurable thresholds and risk."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
from pathlib import Path
from typing import Dict, Iterable

import yaml

from live.agent.core import EmbeddedAgent, SymbolConfig
from live.execution.order_router import OrderRouter
from live.policy.decision import DecisionPolicy, Thresholds
from live.risk.guardrails import RiskGuardrails, RiskLimits
from live.risk.trailing import TrailingManager
from live.state.checkpoint import StateCheckpoint
from ai_trading_bot.config import TrailingConfig

logger = logging.getLogger(__name__)


DEFAULT_THRESHOLDS = {
    "hft": Thresholds(entry=0.55, exit=0.45, cross=0.70, unit_size=0.5),
    "intraday": Thresholds(entry=0.60, exit=0.40, cross=0.75, unit_size=1.0),
    "swing": Thresholds(entry=0.65, exit=0.35, cross=0.80, unit_size=1.0),
}

DEFAULT_RISK_LIMITS = {
    "hft": RiskLimits(leverage=5.0, daily_loss=0.10, atr_stop=2.0, max_position_units=3.0),
    "intraday": RiskLimits(leverage=3.0, daily_loss=0.10, atr_stop=3.0, max_position_units=2.0),
    "swing": RiskLimits(leverage=2.0, daily_loss=0.10, atr_stop=4.0, max_position_units=1.0),
}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} not found.")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _import_api_client(path: str):
    module_path, _, attr = path.partition(":")
    if not attr:
        raise ValueError("API client path must be in module:Class format.")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def build_thresholds(config: dict | None) -> Dict[str, Thresholds]:
    if not config:
        return DEFAULT_THRESHOLDS
    thresholds: Dict[str, Thresholds] = {}
    for regime, values in config.items():
        thresholds[regime] = Thresholds(
            entry=float(values["entry"]),
            exit=float(values["exit"]),
            cross=float(values["cross"]),
            unit_size=float(values.get("unit_size", 1.0)),
        )
    return {**DEFAULT_THRESHOLDS, **thresholds}


def build_risk_limits(config: dict | None) -> Dict[str, RiskLimits]:
    if not config:
        return DEFAULT_RISK_LIMITS
    limits: Dict[str, RiskLimits] = {}
    for key, values in config.items():
        limits[key] = RiskLimits(
            leverage=float(values["leverage"]),
            daily_loss=float(values["daily_loss"]),
            atr_stop=float(values["atr_stop"]),
            max_position_units=float(values.get("max_position_units", 1.0)),
        )
    return {**DEFAULT_RISK_LIMITS, **limits}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the embedded BitMEX agent.")
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration for symbols, thresholds, and risk.")
    parser.add_argument("--api-client", required=True, help="Import path to API client (e.g., live.execution.bitmex:BitmexClient).")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/live_state.json"), help="Checkpoint path.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def build_symbols(entries: Iterable[dict]) -> Dict[str, SymbolConfig]:
    symbols: Dict[str, SymbolConfig] = {}
    for entry in entries:
        cfg = SymbolConfig(symbol=entry["symbol"], regime=entry["regime"])
        symbols[cfg.symbol] = cfg
    return symbols


async def run_agent(args: argparse.Namespace) -> None:
    config = _load_yaml(args.config)
    symbols_cfg = build_symbols(config.get("symbols", []))
    thresholds = build_thresholds(config.get("thresholds"))
    risk_limits = build_risk_limits(config.get("risk_limits"))
    trailing_cfg_dict = (
        config.get("risk", {}).get("trailing")
        or config.get("trailing")
        or {}
    )
    trailing_cfg = TrailingConfig(**trailing_cfg_dict)

    api_client_cls = _import_api_client(args.api_client)
    api_settings = config.get("api", {})
    if hasattr(api_client_cls, "from_config"):
        api_client = api_client_cls.from_config(api_settings)
    else:
        api_client = api_client_cls(**api_settings)

    policy = DecisionPolicy(thresholds)
    risk = RiskGuardrails(risk_limits)
    trailing_manager = TrailingManager(trailing_cfg, risk.positions)
    router = OrderRouter(api_client, trailing_manager=trailing_manager)
    checkpoint = StateCheckpoint(args.checkpoint, risk=risk)
    agent = EmbeddedAgent(symbols_cfg, policy, risk, router, checkpoint, trailing=trailing_manager)

    await agent.start()
    if agent._tasks:
        await asyncio.gather(*agent._tasks.values())


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
