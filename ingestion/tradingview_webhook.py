"""TradingView webhook receiver that forwards alerts to Kafka and DuckDB."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import Any, Dict

from flask import Blueprint, Response, jsonify, request

from ingestion import duckdb_sink
from ingestion.config import TRADINGVIEW
from ingestion.kafka_bridge import KafkaPublisher

logger = logging.getLogger(__name__)

blueprint = Blueprint("tradingview_webhook", __name__)

_publisher = KafkaPublisher(TRADINGVIEW.kafka_topic)
_sink = duckdb_sink.DuckDBSink()
_shutdown_lock = threading.Lock()
_shutdown = False


def _verify_secret(payload: Dict[str, Any]) -> bool:
    if not TRADINGVIEW.shared_secret:
        return True
    return payload.get("secret") == TRADINGVIEW.shared_secret


@blueprint.route("/alerts", methods=["POST"])
def handle_alert() -> Response:
    try:
        data = request.get_json(force=True)
    except Exception:
        logger.warning("TradingView webhook received invalid JSON.")
        return jsonify({"status": "error", "reason": "invalid json"}), 400

    if not isinstance(data, dict):
        return jsonify({"status": "error", "reason": "payload must be object"}), 400

    if not _verify_secret(data):
        logger.warning("TradingView webhook rejected due to secret mismatch.")
        return jsonify({"status": "error", "reason": "unauthorised"}), 401

    symbol = data.get("symbol") or data.get("ticker")
    event_type = data.get("event", "signal")
    ts = data.get("timestamp") or time.time()

    record = {
        "timestamp": ts,
        "symbol": symbol,
        "event": event_type,
        "price": data.get("price"),
        "note": data.get("note"),
        "metadata": {k: v for k, v in data.items() if k not in {"secret", "price", "note", "symbol", "ticker", "event", "timestamp"}},
    }

    _publisher.publish(record)
    _sink.write("tradingview", TRADINGVIEW.kafka_topic, record)
    logger.info("TradingView alert ingested for %s (%s).", symbol, event_type)

    return jsonify({"status": "ok"}), 200


def shutdown() -> None:
    """Flush resources; safe to call multiple times."""
    global _shutdown
    with _shutdown_lock:
        if _shutdown:
            return
        _publisher.close()
        _sink.close()
        _shutdown = True


atexit.register(shutdown)
