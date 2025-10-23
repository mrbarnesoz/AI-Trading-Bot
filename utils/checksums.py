"""Checksum utilities for downloaded archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def md5_path(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_md5(path: Path) -> Path:
    checksum = md5_path(path)
    manifest_path = path.with_suffix(path.suffix + ".md5.json")
    manifest_path.write_text(json.dumps({"path": str(path), "md5": checksum}, indent=2), encoding="utf-8")
    return manifest_path
