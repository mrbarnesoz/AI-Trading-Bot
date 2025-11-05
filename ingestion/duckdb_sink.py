"""DuckDB persistence utilities for ingested feeds."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

import duckdb

from ingestion.config import DUCKDB

logger = logging.getLogger(__name__)


CREATE_SCHEMA_SQL = f"CREATE SCHEMA IF NOT EXISTS {DUCKDB.schema}"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {DUCKDB.schema}.feed_events (
    event_time TIMESTAMP,
    source VARCHAR,
    topic VARCHAR,
    payload JSON
)
"""


class DuckDBSink:
    """Append feed events into DuckDB for training/offline analysis."""

    def __init__(self) -> None:
        self._conn = duckdb.connect(database=str(DUCKDB.database_path), read_only=False)
        self._conn.execute(CREATE_SCHEMA_SQL)
        self._conn.execute(CREATE_TABLE_SQL)
        logger.debug("DuckDB sink initialised at %s", DUCKDB.database_path)

    def write(self, source: str, topic: str, payload: Dict[str, Any]) -> None:
        """Insert a single event row."""
        event_time = datetime.fromtimestamp(payload.get("timestamp", datetime.now(timezone.utc).timestamp()), tz=timezone.utc)
        prepared = json.dumps(payload, separators=(",", ":"))
        self._conn.execute(
            f"INSERT INTO {DUCKDB.schema}.feed_events VALUES (?, ?, ?, ?)",
            [event_time, source, topic, prepared],
        )

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

