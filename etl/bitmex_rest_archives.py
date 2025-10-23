"""Download BitMEX historical archive files and stage them for normalization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import httpx

from utils import io as io_utils
from utils import checksums

logger = logging.getLogger(__name__)


class BitmexArchiveDownloader:
    """Handles BitMEX archive discovery, download, and checksum validation."""

    def __init__(self, base_url: str, output_root: Path, timeout_s: float = 30.0, l2_prefix: str = "orderBookL2") -> None:
        self.base_url = base_url.rstrip("/")
        self.output_root = output_root
        self.timeout_s = timeout_s
        self.l2_prefix = l2_prefix

    def download_daily_archives(self, date_strings: Iterable[str], symbols: Iterable[str], include_l2: bool = True) -> None:
        """Download trade and order book archives for the specified dates/symbols."""
        feeds = ["trade"]
        if include_l2:
            feeds.append(self.l2_prefix)
        for dt in date_strings:
            for symbol in symbols:
                for feed in feeds:
                    success = self._download_single(feed, dt, symbol)
                    if not success:
                        logger.warning("No archive found for %s %s %s", feed, dt, symbol)

    def _download_single(self, feed: str, dt: str, symbol: str) -> bool:
        patterns = [
            f"{self.base_url}/{feed}/{symbol}/{dt}.csv.gz",
            f"{self.base_url}/{feed}/{dt}.csv.gz",
            f"{self.base_url}/{feed}/{symbol}/{dt.replace('-', '')}.csv.gz",
            f"{self.base_url}/{feed}/{dt.replace('-', '')}.csv.gz",
            f"{self.base_url}/{feed}/{symbol}/{dt}.gz",
            f"{self.base_url}/{feed}/{dt}.gz",
        ]
        for url in patterns:
            destination = self.output_root / feed / f"dt={dt}" / Path(url).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                return True
            try:
                with httpx.stream("GET", url, timeout=self.timeout_s) as response:
                    if response.status_code == 404:
                        continue
                    response.raise_for_status()
                    io_utils.write_stream(response, destination)
                checksums.record_md5(destination)
                logger.info("Stored archive %s (%s)", destination, feed)
                return True
            except httpx.HTTPError as exc:
                logger.debug("Failed %s: %s", url, exc)
        return False
