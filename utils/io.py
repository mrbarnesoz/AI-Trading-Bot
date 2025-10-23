"""I/O helpers shared across ETL jobs."""

from __future__ import annotations

import asyncio
import gzip
import io
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import polars as pl


def write_stream(response, destination: Path, chunk_size: int = 1 << 20) -> None:
    """Write a streaming HTTP response to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as fp:
        for chunk in response.iter_bytes(chunk_size):
            fp.write(chunk)


@asynccontextmanager
async def open_async_append(path: Path):
    """Async context manager for appending text to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()

    def open_file() -> io.TextIOWrapper:
        return path.open("a", encoding="utf-8")

    file_obj = await loop.run_in_executor(None, open_file)

    class _AsyncFileWriter:
        def __init__(self, fp: io.TextIOWrapper) -> None:
            self._fp = fp

        async def write(self, data: str) -> None:
            await loop.run_in_executor(None, self._fp.write, data)

        async def flush(self) -> None:
            await loop.run_in_executor(None, self._fp.flush)

    writer = _AsyncFileWriter(file_obj)
    try:
        yield writer
    finally:
        await loop.run_in_executor(None, file_obj.close)


def _read_compressed_json(path: Path) -> pl.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        return pl.read_ndjson(fp)


def read_trade_file(path: Path) -> pl.DataFrame:
    """Read a trade archive (gzipped NDJSON or Parquet)."""
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffixes[-2:] == [".csv", ".gz"]:
        with gzip.open(path, "rt", encoding="utf-8") as fp:
            return pl.read_csv(fp)
    return _read_compressed_json(path)


def read_l2_file(path: Path) -> pl.DataFrame:
    """Read an order book archive (gzipped NDJSON or Parquet)."""
    return read_trade_file(path)


def write_partitioned(df: pl.DataFrame, root: Path, partition_cols: Iterable[str]) -> Path:
    """Write a DataFrame to Parquet partitioned by the supplied columns."""
    root.mkdir(parents=True, exist_ok=True)
    df = df.with_columns(
        [
            df["ts"].dt.truncate("1d").alias("dt"),
        ]
    )
    for keys, group in df.partition_by(list(partition_cols), include_key=True):
        subdir = root
        for key_name, value in zip(partition_cols, keys, strict=False):
            subdir = subdir / f"{key_name}={value}"
        subdir.mkdir(parents=True, exist_ok=True)
        filename = f"part-{hash(tuple(keys)) & 0xFFFF_FFFF:08x}.parquet"
        group.write_parquet(subdir / filename)
    return root


def write_manifest(path: Path, rows: int, extra: Optional[dict] = None) -> Path:
    manifest = {
        "path": str(path),
        "rows": rows,
        "bytes": path.stat().st_size if path.exists() else 0,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(extra)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
