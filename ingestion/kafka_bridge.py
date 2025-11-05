"""Kafka helper utilities with graceful local fallbacks."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ingestion.config import KAFKA

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from confluent_kafka import Producer  # type: ignore
except Exception:  # pragma: no cover - used when confluent_kafka is unavailable
    Producer = None  # type: ignore


@dataclass
class LocalQueueMessage:
    """Fallback envelope written when Kafka is not available."""

    topic: str
    payload: Dict[str, Any]
    timestamp: float


class KafkaPublisher:
    """Publish JSON messages to Kafka with a local file fallback."""

    def __init__(self, topic: str, delivery_timeout_ms: int = 5_000) -> None:
        self.topic = topic
        self._delivery_lock = threading.Lock()
        self._fallback_path = Path("logs") / f"{topic.replace('.', '_')}_fallback.jsonl"
        self._local_queue: queue.Queue[LocalQueueMessage] = queue.Queue()
        self._publisher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if Producer:
            config = {
                "bootstrap.servers": KAFKA.bootstrap_servers,
                "socket.timeout.ms": delivery_timeout_ms,
                "message.timeout.ms": delivery_timeout_ms,
                "security.protocol": KAFKA.security_protocol,
            }
            if KAFKA.ssl_cafile:
                config["ssl.ca.location"] = KAFKA.ssl_cafile
            if KAFKA.ssl_certfile:
                config["ssl.certificate.location"] = KAFKA.ssl_certfile
            if KAFKA.ssl_keyfile:
                config["ssl.key.location"] = KAFKA.ssl_keyfile
            try:
                self._producer = Producer(config)  # type: ignore[call-arg]
                logger.info("Kafka publisher initialised for topic %s", topic)
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning("Kafka producer init failed (%s); using local queue fallback.", exc)
                self._producer = None
        else:
            logger.info("confluent_kafka not installed; using local queue fallback.")
            self._producer = None

        if not self._producer:
            self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
            self._publisher_thread = threading.Thread(target=self._drain_local_queue, daemon=True)
            self._publisher_thread.start()

    def publish(self, message: Dict[str, Any]) -> None:
        """Serialise and publish the message."""
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        timestamp = time.time()

        if self._producer:
            try:
                self._producer.produce(self.topic, payload)
                self._producer.poll(0)
                return
            except Exception as exc:  # pragma: no cover - fallback when Kafka unreachable
                logger.warning("Kafka publish failed (%s); queueing to local fallback.", exc)

        self._local_queue.put(LocalQueueMessage(self.topic, message, timestamp))

    def close(self) -> None:
        """Flush any pending data."""
        if self._producer:
            with self._delivery_lock:
                self._producer.flush(5.0)
        else:
            self._stop_event.set()
            if self._publisher_thread:
                self._publisher_thread.join(timeout=2.0)

    def _drain_local_queue(self) -> None:
        """Write queued records to disk for later replay."""
        while not self._stop_event.is_set():
            try:
                item = self._local_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            record = {
                "topic": item.topic,
                "payload": item.payload,
                "timestamp": item.timestamp,
            }
            try:
                with self._fallback_path.open("a", encoding="utf-8") as handle:
                    json.dump(record, handle)
                    handle.write("\n")
            except OSError as exc:
                logger.error("Failed to write fallback record: %s", exc)


def replay_fallback(path: Path, topics: Iterable[str]) -> Iterable[Dict[str, Any]]:
    """Yield fallback messages for the given topics."""
    if not path.exists():
        return []
    matches = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("topic") in topics:
                    matches.append(record)
    except OSError as exc:
        logger.error("Unable to replay fallback data: %s", exc)
    return matches

