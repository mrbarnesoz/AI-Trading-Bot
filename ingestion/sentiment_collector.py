"""Poll a sentiment source and publish scores to Kafka / DuckDB."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx

from ingestion import duckdb_sink
from ingestion.config import SENTIMENT
from ingestion.kafka_bridge import KafkaPublisher

logger = logging.getLogger(__name__)

_seen_items: set[str] = set()


def _headline_key(item: Dict[str, Any]) -> str:
    return str(
        item.get("id")
        or item.get("url")
        or item.get("headline")
        or item.get("title")
        or time.time()
    )


def _load_headlines(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        logger.warning("Sentiment source %s missing.", path)
        return []
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Sentiment source %s invalid JSON: %s", path, exc)
        return []
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        items = content.get("items") or content.get("results") or []
        return list(items)
    return []


def _fetch_newsapi_headlines() -> List[Dict[str, Any]]:
    if not SENTIMENT.api_url or not SENTIMENT.api_key:
        logger.warning("Sentiment provider newsapi requires SENTIMENT_API_URL and SENTIMENT_API_KEY.")
        return []
    params = {
        "q": SENTIMENT.query,
        "language": SENTIMENT.language,
        "pageSize": SENTIMENT.max_items,
        "sortBy": "publishedAt",
    }
    headers = {"X-Api-Key": SENTIMENT.api_key}
    try:
        with httpx.Client(timeout=SENTIMENT.timeout_seconds) as client:
            response = client.get(SENTIMENT.api_url, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("Failed to fetch sentiment from NewsAPI: %s", exc)
        return []

    articles = payload.get("articles", [])
    records: List[Dict[str, Any]] = []
    for article in articles:
        records.append(
            {
                "id": article.get("url"),
                "headline": article.get("title"),
                "summary": article.get("description"),
                "timestamp": article.get("publishedAt"),
                "symbols": [],
                "source": article.get("source", {}).get("name", "newsapi"),
                "url": article.get("url"),
            }
        )
    return records


def _collect_headlines() -> List[Dict[str, Any]]:
    provider = SENTIMENT.provider.lower().strip()
    records: List[Dict[str, Any]] = []

    if provider == "newsapi":
        records = _fetch_newsapi_headlines()
    elif provider not in ("file", "", "local"):
        logger.warning("Unknown sentiment provider '%s'; falling back to file source.", provider)

    if not records:
        records = _load_headlines(SENTIMENT.source_path)

    return records


def _score_headline(text: str, summary: str | None = None) -> float:
    """Very lightweight lexicon-based scorer."""
    text_lower = (text or "").lower()
    if summary:
        text_lower += " " + summary.lower()
    positive = sum(text_lower.count(word) for word in ("bull", "surge", "gain", "beat", "optimistic", "buy"))
    negative = sum(text_lower.count(word) for word in ("bear", "drop", "loss", "miss", "fear", "sell"))
    total = max(len(text_lower.split()), 1)
    return (positive - negative) / total


def run_once(publisher: KafkaPublisher, sink: duckdb_sink.DuckDBSink) -> int:
    """Process the source once; returns number of records dispatched."""
    records = 0
    for item in _collect_headlines():
        headline = str(item.get("headline") or item.get("title") or "")
        if not headline:
            continue
        key = _headline_key(item)
        if key in _seen_items:
            continue
        _seen_items.add(key)
        summary = item.get("summary")
        score = _score_headline(headline, summary)
        timestamp = item.get("timestamp") or time.time()
        payload = {
            "timestamp": timestamp,
            "headline": headline,
            "summary": summary,
            "symbols": item.get("symbols", []),
            "score": score,
            "source": item.get("source", SENTIMENT.provider),
            "meta": {k: v for k, v in item.items() if k not in {"headline", "summary", "symbols", "timestamp"}},
        }
        publisher.publish(payload)
        sink.write("sentiment", SENTIMENT.kafka_topic, payload)
        records += 1
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    publisher = KafkaPublisher(SENTIMENT.kafka_topic)
    sink = duckdb_sink.DuckDBSink()
    try:
        while True:
            count = run_once(publisher, sink)
            logger.info("Sentiment collector dispatched %s records.", count)
            time.sleep(SENTIMENT.poll_interval_seconds)
    except KeyboardInterrupt:
        logger.info("Sentiment collector shutting down.")
    finally:
        publisher.close()
        sink.close()


if __name__ == "__main__":
    main()

