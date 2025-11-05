"""Market data feed abstractions for live trading."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency during tests
    import websocket  # type: ignore[import]
except ImportError:  # pragma: no cover - fallback for environments without websocket-client
    websocket = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

TradeCallback = Callable[["TradeEvent"], None]
BarCallback = Callable[["BarEvent"], None]


@dataclass(frozen=True)
class TradeEvent:
    """Atomic trade event as reported by the venue."""

    symbol: str
    price: float
    size: float
    timestamp: datetime


@dataclass(frozen=True)
class BarEvent:
    """Aggregated OHLCV bar over a fixed interval."""

    symbol: str
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataFeed:
    """Abstract market data feed."""

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def register_trade_callback(self, callback: TradeCallback) -> None:
        raise NotImplementedError

    def register_bar_callback(self, callback: BarCallback) -> None:
        raise NotImplementedError


class BitmexWebsocketDataFeed(DataFeed):
    """BitMEX websocket feed for trades aggregated into bars."""

    _BITMEX_ENDPOINT = "wss://stream.bitmex.com/realtime"

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        bar_interval_seconds: int = 60,
        endpoint: str | None = None,
        max_reconnect_delay: float = 30.0,
    ) -> None:
        self.symbols = list({symbol.upper() for symbol in symbols})
        if not self.symbols:
            raise ValueError("BitmexWebsocketDataFeed requires at least one symbol.")
        self.bar_interval_seconds = max(10, int(bar_interval_seconds))
        self.endpoint = endpoint or self._BITMEX_ENDPOINT
        self.max_reconnect_delay = max_reconnect_delay

        self._trade_callbacks: List[TradeCallback] = []
        self._bar_callbacks: List[BarCallback] = []

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._current_bars: Dict[str, Dict[str, float]] = {}
        self._bar_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

    def register_trade_callback(self, callback: TradeCallback) -> None:
        self._trade_callbacks.append(callback)

    def register_bar_callback(self, callback: BarCallback) -> None:
        self._bar_callbacks.append(callback)

    def start(self) -> None:
        if websocket is None:
            raise RuntimeError(
                "websocket-client is required for BitMEX live feeds. Install with 'pip install websocket-client'."
            )
        if self._thread and self._thread.is_alive():
            logger.debug("BitMEX data feed already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_forever, name="bitmex-feed", daemon=True)
        self._thread.start()
        logger.info("BitMEX websocket data feed started for %s", ", ".join(self.symbols))

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error while closing BitMEX websocket.")
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("BitMEX websocket data feed stopped.")

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _run_forever(self) -> None:
        delay = 1.0
        while not self._stop_event.is_set():
            try:
                self._connect()
                delay = 1.0  # reset on successful run
            except Exception as exc:  # pragma: no cover - runtime resilience
                logger.warning("BitMEX feed error: %s", exc, exc_info=True)
            if self._stop_event.is_set():
                break
            time.sleep(delay)
            delay = min(delay * 2, self.max_reconnect_delay)

    def _connect(self) -> None:
        params = ",".join(f"trade:{symbol}" for symbol in self.symbols)
        url = f"{self.endpoint}?subscribe={params}"

        def _on_open(ws: websocket.WebSocketApp) -> None:
            logger.info("BitMEX websocket connected.")
            self._ws = ws

        def _on_message(ws: websocket.WebSocketApp, message: str) -> None:  # pragma: no cover - network
            self._handle_message(message)

        def _on_error(ws: websocket.WebSocketApp, error: Exception) -> None:  # pragma: no cover - network
            if not self._stop_event.is_set():
                logger.error("BitMEX websocket error: %s", error)

        def _on_close(ws: websocket.WebSocketApp, code: int, msg: str) -> None:  # pragma: no cover - network
            if not self._stop_event.is_set():
                logger.info("BitMEX websocket closed (%s): %s", code, msg)

        self._ws = websocket.WebSocketApp(
            url,
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )
        self._ws.run_forever(sslopt={"check_hostname": False})

    def _handle_message(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:  # pragma: no cover - network
            logger.debug("Ignoring non-JSON websocket payload: %s", message)
            return

        if payload.get("table") != "trade":
            return

        for trade in payload.get("data", []):
            symbol = trade.get("symbol")
            if symbol not in self.symbols:
                continue
            timestamp = self._parse_timestamp(trade.get("timestamp"))
            price = float(trade.get("price", 0.0))
            size = float(trade.get("size", 0.0))
            event = TradeEvent(symbol=symbol, price=price, size=size, timestamp=timestamp)
            self._emit_trade(event)
            self._update_bar(event)

    def _parse_timestamp(self, value: Optional[str]) -> datetime:
        if not value:
            return datetime.now(tz=timezone.utc)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:  # pragma: no cover - defensive
            return datetime.now(tz=timezone.utc)

    def _emit_trade(self, event: TradeEvent) -> None:
        for callback in list(self._trade_callbacks):
            try:
                callback(event)
            except Exception:  # pragma: no cover - callback safety
                logger.exception("Trade callback error for %s", event.symbol)

    def _update_bar(self, event: TradeEvent) -> None:
        state = self._current_bars.get(event.symbol)
        interval = self.bar_interval_seconds
        bar_start_epoch = int(event.timestamp.timestamp() // interval * interval)
        bar_start = datetime.fromtimestamp(bar_start_epoch, tz=timezone.utc)
        bar_end = bar_start + timedelta(seconds=interval)

        with self._bar_locks[event.symbol]:
            if state and event.timestamp >= state["end"]:
                bar_event = BarEvent(
                    symbol=event.symbol,
                    start=state["start"],
                    end=state["end"],
                    open=state["open"],
                    high=state["high"],
                    low=state["low"],
                    close=state["close"],
                    volume=state["volume"],
                )
                self._emit_bar(bar_event)
                state = None

            if not state:
                state = {
                    "start": bar_start,
                    "end": bar_start + timedelta(seconds=interval),
                    "open": event.price,
                    "high": event.price,
                    "low": event.price,
                    "close": event.price,
                    "volume": event.size,
                }
            else:
                state["high"] = max(state["high"], event.price)
                state["low"] = min(state["low"], event.price)
                state["close"] = event.price
                state["volume"] += event.size

            self._current_bars[event.symbol] = state

    def _emit_bar(self, bar: BarEvent) -> None:
        for callback in list(self._bar_callbacks):
            try:
                callback(bar)
            except Exception:  # pragma: no cover - callback safety
                logger.exception("Bar callback error for %s", bar.symbol)
