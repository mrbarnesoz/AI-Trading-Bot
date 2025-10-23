"""CLI entrypoint to build L2 snapshots from normalized updates."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from etl.orderbook_snapshots import SnapshotConfig, build_snapshots, snapshots_to_parquet

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build L2 snapshots from normalized updates.")
    parser.add_argument("--updates-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cadence-ms", type=int, default=250)
    parser.add_argument("--burst-on-change", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SnapshotConfig(cadence_ms=args.cadence_ms, burst_on_best_change=args.burst_on_change)
    update_files = sorted(args.updates_root.rglob("*.parquet"))
    for update_file in update_files:
        logger.info("Processing updates %s", update_file)
        updates = pl.read_parquet(update_file)
        snapshots = build_snapshots(updates, config)
        if snapshots.height == 0:
            continue
        relative = update_file.relative_to(args.updates_root)
        output_path = args.output_root / relative
        snapshots_to_parquet(snapshots, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
