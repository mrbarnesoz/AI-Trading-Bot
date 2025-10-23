"""WebSocket recorder for BitMEX real-time streams."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Iterable

import websockets

from utils import io as io_utils
from utils.time import utc_now

logger = logging.getLogger(__name__)


class BitmexWebSocketRecorder:
    """Persist BitMEX WebSocket messages to newline-delimited JSON files."""

    def __init__(self, ws_url: str, channels: Iterable[str], output_dir: Path) -> None:
        self.ws_url = ws_url
        self.channels = list(channels)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> None:
        async with websockets.connect(self.ws_url, ping_interval=5) as ws:
            await ws.send(json.dumps({"op": "subscribe", "args": self.channels}))
            logger.info("Subscribed to channels: %s", ",".join(self.channels))
            async for message in ws:
                await self._persist_message(message)

    async def _persist_message(self, message: str) -> None:
        timestamp = utc_now().strftime("%Y%m%d%H%M%S")
        daily_dir = self.output_dir / f"dt={timestamp[:8]}"
        daily_dir.mkdir(parents=True, exist_ok=True)
        file_path = daily_dir / f"bitmex_ws_{timestamp}.jsonl"
        async with io_utils.open_async_append(file_path) as writer:
            await writer.write(message + "\n")
        logger.debug("Recorded message to %s", file_path)
