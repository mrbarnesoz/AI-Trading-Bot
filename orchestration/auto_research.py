"""Automated strategy research utilities for regime-aware selection."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

from ai_trading_bot.config import AppConfig, load_config
from ai_trading_bot.pipeline import execute_backtest

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "configs"
DEFAULT_CONFIG = BASE_DIR / "config.yaml"
AUTO_RESULTS_DIR = BASE_DIR / "results" / "auto"


@dataclass
class StrategyCandidate:
    config_path: Path
    symbol: str
    timeframe: str
    strategy: str


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


def discover_candidates(
    *,
    symbols: Optional[Sequence[str]] = None,
    timeframes: Optional[Sequence[str]] = None,
    strategies: Optional[Sequence[str]] = None,
) -> List[StrategyCandidate]:
    """Identify strategy configs matching optional filters."""
    symbols_set = {s.upper() for s in symbols or []}
    timeframes_set = {tf.lower() for tf in timeframes or []}
    strategies_set = {s.lower() for s in strategies or []}

    candidates: List[StrategyCandidate] = []
    search_dirs = [CONFIG_DIR] if CONFIG_DIR.exists() else []
    for directory in search_dirs:
        for path in directory.glob("*.y*ml"):
            stem = path.stem
            parts = stem.split("_")
            symbol = parts[0].upper() if parts else "XBTUSD"
            timeframe = parts[1].lower() if len(parts) >= 2 else ""
            strategy_name = "_".join(parts[2:]) if len(parts) >= 3 else parts[-1] if parts else stem

            if symbols_set and symbol not in symbols_set:
                continue
            if timeframes_set and timeframe not in timeframes_set:
                continue
            if strategies_set and strategy_name.lower() not in strategies_set:
                continue

            candidates.append(
                StrategyCandidate(
                    config_path=path,
                    symbol=symbol,
                    timeframe=timeframe or "unknown",
                    strategy=strategy_name,
                )
            )

    if not candidates and DEFAULT_CONFIG.exists():
        cfg = load_config(DEFAULT_CONFIG)
        candidates.append(
            StrategyCandidate(
                config_path=DEFAULT_CONFIG,
                symbol=(cfg.data.symbol or "XBTUSD").upper(),
                timeframe=(cfg.data.interval or "1h").lower(),
                strategy="default",
            )
        )
    return candidates


def _prepare_active_config(config_path: Path) -> AppConfig:
    try:
        return load_config(str(config_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load config {config_path}: {exc}") from exc


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def run_auto_research(
    *,
    symbols: Optional[Sequence[str]] = None,
    timeframes: Optional[Sequence[str]] = None,
    strategies: Optional[Sequence[str]] = None,
    min_trades: int = 100,
    top_k: int = 3,
) -> Path:
    """Run backtests for matching configs and summarise best candidates per symbol/timeframe."""

    candidates = discover_candidates(symbols=symbols, timeframes=timeframes, strategies=strategies)
    if not candidates:
        raise RuntimeError("No strategy candidates discovered. Ensure configs/ contains definitions.")

    AUTO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)

    for candidate in candidates:
        logger.info(
            "Auto research evaluating %s (symbol=%s timeframe=%s strategy=%s)",
            candidate.config_path,
            candidate.symbol,
            candidate.timeframe,
            candidate.strategy,
        )
        config = _prepare_active_config(candidate.config_path)
        strategy_output, result, metadata = execute_backtest(
            config,
            force_download=False,
            long_threshold=None,
            short_threshold=None,
            verbose=False,
            enforce_gates=False,
        )
        summary = dict(result.summary)
        stats = {
            "config": str(candidate.config_path),
            "symbol": candidate.symbol,
            "timeframe": candidate.timeframe,
            "strategy": candidate.strategy,
            "calc_sharpe": _safe_float(summary.get("calc_sharpe")),
            "total_return": _safe_float(summary.get("total_return")),
            "max_drawdown": _safe_float(summary.get("max_drawdown")),
            "trades_count": int(summary.get("trades_count", 0) or 0),
            "expectancy": _safe_float(summary.get("expectancy_after_costs")),
            "win_rate": _safe_float(summary.get("win_rate")),
            "metadata": metadata,
            "multi_strategy_regimes": metadata.get("multi_strategy_regimes") if isinstance(metadata, dict) else None,
            "strategy_modules": metadata.get("strategy_modules") if isinstance(metadata, dict) else None,
        }
        stats["meets_trade_gate"] = stats["trades_count"] >= min_trades
        summary_rows.append(stats)
        grouped[(candidate.symbol, candidate.timeframe)].append(stats)

    selections: List[Dict[str, object]] = []
    for (symbol, timeframe), rows in grouped.items():
        ranked = sorted(rows, key=lambda row: (row["meets_trade_gate"], row["calc_sharpe"], row["total_return"]), reverse=True)
        selected = ranked[:max(1, top_k)]
        selections.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "candidates": selected,
                "best": selected[0] if selected else None,
            }
        )

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "min_trades": min_trades,
        "top_k": top_k,
        "inputs": {
            "symbols": symbols,
            "timeframes": timeframes,
            "strategies": strategies,
        },
        "results": summary_rows,
        "selections": selections,
    }
    output_path = AUTO_RESULTS_DIR / f"auto_research-{_timestamp()}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Auto research summary written to %s", output_path)
    return output_path
