"""Replay BitMEX L2 updates into fixed-cadence snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl

from utils.io import write_manifest


@dataclass
class SnapshotConfig:
    cadence_ms: int = 250
    burst_on_best_change: bool = True
    depth_levels: int = 5


def build_snapshots(updates: pl.DataFrame, config: SnapshotConfig) -> pl.DataFrame:
    """Replay updates per symbol and emit snapshots."""
    updates = updates.sort("ts")
    result_frames: List[pl.DataFrame] = []
    for symbol, group in updates.groupby("symbol"):
        result_frames.append(_replay_symbol(symbol, group, config))
    if not result_frames:
        return pl.DataFrame()
    return pl.concat(result_frames)


def _replay_symbol(symbol: str, updates: pl.DataFrame, config: SnapshotConfig) -> pl.DataFrame:
    rows = updates.to_dicts()
    bids: Dict[float, float] = {}
    asks: Dict[float, float] = {}
    level_price: Dict[Tuple[str, int], float] = {}
    snapshots: List[dict] = []
    if not rows:
        return pl.DataFrame()
    cadence = timedelta(milliseconds=config.cadence_ms)
    current_time = rows[0]["ts"]
    next_snapshot = current_time + cadence
    last_best = (None, None)
    for row in rows:
        ts = row["ts"]
        side = row["side"]
        action = row["action"]
        level_id = row.get("level_id")
        price = row.get("price")
        size = row.get("size")
        book = bids if side == "Bid" else asks
        key = (side, int(level_id) if level_id is not None else 0)

        if action == "delete":
            price = level_price.pop(key, None)
            if price is not None:
                book.pop(price, None)
        elif action in ("insert", "update"):
            if price is None:
                price = level_price.get(key)
            if price is not None:
                level_price[key] = price
                if size is None or size <= 0:
                    book.pop(price, None)
                else:
                    book[price] = float(size)

        best_bid = max(bids.keys()) if bids else None
        best_ask = min(asks.keys()) if asks else None

        if (
            config.burst_on_best_change
            and (best_bid, best_ask) != last_best
            and best_bid is not None
            and best_ask is not None
        ):
            snapshots.append(_build_snapshot(symbol, ts, bids, asks, config.depth_levels))
            last_best = (best_bid, best_ask)

        while ts >= next_snapshot:
            if best_bid is not None and best_ask is not None:
                snapshots.append(_build_snapshot(symbol, next_snapshot, bids, asks, config.depth_levels))
            next_snapshot += cadence

    if snapshots:
        ingest_ts = datetime.now(timezone.utc)
        for snap in snapshots:
            snap["ingest_ts"] = ingest_ts
        return pl.DataFrame(snapshots)
    return pl.DataFrame()


def _build_snapshot(symbol: str, ts: datetime, bids: Dict[float, float], asks: Dict[float, float], levels: int) -> dict:
    bid_levels = sorted(bids.items(), key=lambda kv: kv[0], reverse=True)[:levels]
    ask_levels = sorted(asks.items(), key=lambda kv: kv[0])[:levels]
    snapshot = {"ts": ts, "symbol": symbol}
    if bid_levels:
        snapshot["bbp"] = bid_levels[0][0]
    if ask_levels:
        snapshot["bap"] = ask_levels[0][0]
    if "bbp" in snapshot and "bap" in snapshot:
        snapshot["spread"] = snapshot["bap"] - snapshot["bbp"]
        snapshot["mid"] = snapshot["bbp"] + snapshot["spread"] / 2
    else:
        snapshot["spread"] = None
        snapshot["mid"] = None
    total_bid = 0.0
    total_ask = 0.0
    for idx, (price, size) in enumerate(bid_levels, start=1):
        snapshot[f"bid_px_{idx}"] = price
        snapshot[f"bid_sz_{idx}"] = size
        total_bid += size
    for idx in range(len(bid_levels) + 1, levels + 1):
        snapshot[f"bid_px_{idx}"] = None
        snapshot[f"bid_sz_{idx}"] = 0.0
    for idx, (price, size) in enumerate(ask_levels, start=1):
        snapshot[f"ask_px_{idx}"] = price
        snapshot[f"ask_sz_{idx}"] = size
        total_ask += size
    for idx in range(len(ask_levels) + 1, levels + 1):
        snapshot[f"ask_px_{idx}"] = None
        snapshot[f"ask_sz_{idx}"] = 0.0
    snapshot["total_bid_sz_1_5"] = total_bid
    snapshot["total_ask_sz_1_5"] = total_ask
    return snapshot


def snapshots_to_parquet(snapshots: pl.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.write_parquet(output_path)
    write_manifest(output_path, rows=snapshots.height)
    return output_path
