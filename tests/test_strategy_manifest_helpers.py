import os
import textwrap

os.environ.setdefault("DUCKDB_PATH", ":memory:")

from tradingbotui import strategy_manifest


def test_get_strategy_manifest_reads_yaml(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        textwrap.dedent(
            """
            swing_trading:
              label: "Swing Trading"
              description: "Medium-term trades."
              timeframes: ["4h", "1d"]
              default: "4h"
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(strategy_manifest, "MANIFEST_PATH", manifest_path)
    strategy_manifest.get_strategy_manifest.cache_clear()
    try:
        manifest = strategy_manifest.get_strategy_manifest()
        assert "swing_trading" in manifest
        entry = manifest["swing_trading"]
        assert entry["label"] == "Swing Trading"
        assert entry["timeframes"] == ["4h", "1d"]
        assert entry["default"] == "4h"
    finally:
        strategy_manifest.get_strategy_manifest.cache_clear()


def test_get_strategy_manifest_missing_file(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr(strategy_manifest, "MANIFEST_PATH", missing_path)
    strategy_manifest.get_strategy_manifest.cache_clear()
    try:
        assert strategy_manifest.get_strategy_manifest() == {}
    finally:
        strategy_manifest.get_strategy_manifest.cache_clear()
