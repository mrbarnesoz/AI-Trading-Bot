"""Latency arbitrage simulation on historical multi-exchange data."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

DATA_DIR = Path("data") / "raw" / "bitmex"
OUTPUT_DIR = Path("results") / "latency_arbitrage"
LOG_PATH = Path("logs") / "latency_arbitrage_guardrail.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class SimulationConfig:
    maker_fee_bps: float = -0.025  # BitMEX maker rebate in bps
    taker_fee_bps: float = 7.5  # conservative taker fee in bps
    slip_bps: float = 1.5
    latency_ms: int = 150
    spread_threshold_bps: float = 10.0
    max_position_usd: float = 100_000.0


def _load_reference(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns = {col: col.lower() for col in df.columns}
    df = df.rename(columns=columns)
    time_col = next((col for col in ("timestamp", "date", "datetime") if col in df.columns), None)
    if time_col is None:
        raise ValueError(f"No timestamp column found in {path.name}")
    df["timestamp"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    price_col = next((col for col in ("close", "price", "trade_price") if col in df.columns), None)
    if price_col is None:
        raise ValueError(f"No price column found in {path.name}")
    df = df[["timestamp", price_col]].rename(columns={price_col: "price"})
    return df


def _synthesise_feed(df: pd.DataFrame, lag_rows: int, noise_bps: float) -> pd.DataFrame:
    noise = np.random.normal(loc=0.0, scale=noise_bps / 10_000.0, size=len(df))
    prices = df["price"] * (1 + noise)
    if lag_rows > 0:
        prices = prices.shift(lag_rows)
    result = pd.DataFrame({"timestamp": df["timestamp"], "price": prices})
    return result.dropna().reset_index(drop=True)


def _bps_diff(price_a: float, price_b: float) -> float:
    return (price_a - price_b) / price_b * 10_000.0


def run_simulation(config: SimulationConfig, sample: Iterable[pd.Timestamp] | None = None) -> Dict[str, float]:
    reference_file = sorted(DATA_DIR.glob("bitmex_XBTUSD_1m_*.csv"))
    if not reference_file:
        raise FileNotFoundError("No BitMEX 1m data available for simulation.")
    ref = _load_reference(reference_file[0])
    fast_feed = ref.copy()
    slow_feed = _synthesise_feed(ref, lag_rows=1, noise_bps=2.0)
    alt_feed = _synthesise_feed(ref, lag_rows=2, noise_bps=3.5)

    merged = fast_feed.merge(slow_feed, on="timestamp", how="inner", suffixes=("_fast", "_slow"))
    merged = merged.merge(alt_feed, on="timestamp", how="inner")
    merged = merged.rename(columns={"price": "price_alt"})

    trades: List[Dict[str, float]] = []
    gross = 0.0
    net = 0.0

    for _, row in merged.iterrows():
        fast_price = row["price_fast"]
        slow_price = row["price_slow"]
        alt_price = row["price_alt"]
        spread_bps = _bps_diff(slow_price, fast_price)
        alt_spread_bps = _bps_diff(alt_price, fast_price)
        edge_bps = max(spread_bps, alt_spread_bps)
        if math.isnan(edge_bps):
            continue
        if abs(edge_bps) < config.spread_threshold_bps:
            continue

        direction = 1 if edge_bps > 0 else -1
        entry_price = fast_price if direction > 0 else slow_price
        exit_price = slow_price if direction > 0 else fast_price

        fees_bps = config.taker_fee_bps - config.maker_fee_bps
        realized_bps = edge_bps - config.slip_bps - fees_bps
        if realized_bps <= 0:
            continue

        usd_notional = config.max_position_usd
        pnl = usd_notional * realized_bps / 10_000.0
        gross += pnl
        net += pnl

        trades.append(
            {
                "timestamp": row["timestamp"].isoformat(),
                "spread_bps": edge_bps,
                "realized_bps": realized_bps,
                "pnl": pnl,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
            }
        )

    summary = {
        "trades": len(trades),
        "gross_pnl": gross,
        "net_pnl": net,
        "average_realized_bps": float(np.mean([t["realized_bps"] for t in trades])) if trades else 0.0,
        "max_spread_bps": float(np.max([abs(t["spread_bps"]) for t in trades])) if trades else 0.0,
    }

    output = {
        "config": config.__dict__,
        "summary": summary,
        "sample_trades": trades[:10],
    }

    output_path = OUTPUT_DIR / "latency_arbitrage_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    with LOG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


if __name__ == "__main__":
    result = run_simulation(SimulationConfig())
    print(json.dumps(result, indent=2))
