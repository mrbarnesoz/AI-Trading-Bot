#!/usr/bin/env python
"""Utility to remove cached BitMEX datasets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


RAW_DIR = Path("data/raw")
ENGINEERED_DIR = Path("data/engineered")


def _delete_matching(paths: Iterable[Path], dry_run: bool) -> int:
    count = 0
    for path in paths:
        if dry_run:
            print(f"[DRY RUN] Would remove {path}")
        else:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"Removed {path}")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean cached BitMEX data (raw and engineered).")
    parser.add_argument("--symbol", help="Filter cache files by symbol (e.g. XBTUSD).")
    parser.add_argument("--all", action="store_true", help="Remove entire raw and engineered directories.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting.")
    args = parser.parse_args()

    if args.all:
        targets = [RAW_DIR, ENGINEERED_DIR]
        removed = _delete_matching([path for path in targets if path.exists()], args.dry_run)
        if removed == 0:
            print("No cache directories found to remove.")
        return

    if not args.symbol:
        parser.error("Specify --symbol or use --all to remove every cached dataset.")

    pattern = f"*{args.symbol.upper()}*"
    candidates = []
    if RAW_DIR.exists():
        candidates.extend(RAW_DIR.glob(pattern))
    if ENGINEERED_DIR.exists():
        candidates.extend(ENGINEERED_DIR.glob(pattern))

    if not candidates:
        print(f"No cached files found matching {pattern!r}.")
        return

    _delete_matching(candidates, args.dry_run)


if __name__ == "__main__":
    main()
