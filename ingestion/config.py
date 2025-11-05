"""Configuration helpers for ingestion services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KafkaSettings:
    """Kafka connection parameters."""

    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    security_protocol: str = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    ssl_cafile: str | None = os.getenv("KAFKA_SSL_CAFILE")
    ssl_certfile: str | None = os.getenv("KAFKA_SSL_CERTFILE")
    ssl_keyfile: str | None = os.getenv("KAFKA_SSL_KEYFILE")


@dataclass(frozen=True)
class DuckDBSettings:
    """DuckDB persistence parameters."""

    database_path: Path = Path(
        os.getenv("DUCKDB_PATH", Path(os.getcwd()) / "warehouse.duckdb")
    )
    schema: str = os.getenv("DUCKDB_SCHEMA", "raw_feeds")


@dataclass(frozen=True)
class TradingViewSettings:
    """TradingView webhook configuration."""

    shared_secret: str = os.getenv("TRADINGVIEW_SHARED_SECRET", "")
    kafka_topic: str = os.getenv("TRADINGVIEW_TOPIC", "tradingview.alerts")


@dataclass(frozen=True)
class SentimentSettings:
    """Sentiment feed configuration."""

    source_path: Path = Path(
        os.getenv("SENTIMENT_SOURCE", Path(os.getcwd()) / "data" / "sentiment" / "headlines.json")
    )
    kafka_topic: str = os.getenv("SENTIMENT_TOPIC", "sentiment.signals")
    poll_interval_seconds: int = int(os.getenv("SENTIMENT_POLL_INTERVAL", "60"))
    provider: str = os.getenv("SENTIMENT_PROVIDER", "file")
    api_url: str = os.getenv("SENTIMENT_API_URL", "")
    api_key: str = os.getenv("SENTIMENT_API_KEY", "")
    query: str = os.getenv("SENTIMENT_QUERY", "bitcoin OR crypto")
    language: str = os.getenv("SENTIMENT_LANGUAGE", "en")
    max_items: int = int(os.getenv("SENTIMENT_MAX_ITEMS", "50"))
    timeout_seconds: float = float(os.getenv("SENTIMENT_TIMEOUT", "5"))


KAFKA = KafkaSettings()
DUCKDB = DuckDBSettings()
TRADINGVIEW = TradingViewSettings()
SENTIMENT = SentimentSettings()
