"""Notification helpers using Apprise and Prefect blocks."""

from __future__ import annotations

import os
from typing import Iterable, List

from apprise import Apprise

from orchestration.block_helpers import load_secret_block


def _parse_urls(value: str) -> List[str]:
    return [u.strip() for u in value.split(",") if u.strip()]


def get_notification_urls() -> List[str]:
    env_urls = os.getenv("APPRISE_URLS")
    if env_urls:
        return _parse_urls(env_urls)
    secret_value = load_secret_block("alert-webhooks")
    if secret_value:
        if isinstance(secret_value, str):
            return _parse_urls(secret_value)
        if isinstance(secret_value, (list, tuple)):
            return [str(u).strip() for u in secret_value if str(u).strip()]
    return []


def send_notification(message: str, title: str = "BitMEX Pipeline Alert") -> bool:
    urls = get_notification_urls()
    if not urls:
        return False
    app = Apprise()
    for url in urls:
        app.add(url)
    return app.notify(title=title, body=message)
