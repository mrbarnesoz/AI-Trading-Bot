"""Utilities for building temp config files with overrides."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import yaml

from utils.config import load_yaml


def build_config(base_path: Path, override: Dict) -> Path:
    config = load_yaml(base_path)
    if override:
        for key, value in override.items():
            config[key] = value
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(config, tmp)
        tmp_path = Path(tmp.name)
    return tmp_path
