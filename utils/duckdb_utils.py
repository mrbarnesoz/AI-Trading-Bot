"""DuckDB helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb


def ensure_views(db_path: Path, view_statements: Iterable[str]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as conn:
        for statement in view_statements:
            conn.execute(statement)
