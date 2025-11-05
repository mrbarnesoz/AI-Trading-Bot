"""Celery application factory for the trading bot UI."""

from __future__ import annotations

import os

try:
    from celery import Celery  # type: ignore
    _CELERY_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback when celery is unavailable
    _CELERY_AVAILABLE = False

    class _DummyConf(dict):
        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)

    class Celery:  # type: ignore[override]
        """Minimal stand-in when Celery is not installed."""

        def __init__(self, *args, **kwargs):
            self.conf = _DummyConf()

        def task(self, *args, **kwargs):  # noqa: D401 - mimic Celery API
            def decorator(func):
                return func

            return decorator

        def __getattr__(self, item):
            raise AttributeError(
                "Celery is not installed; install the 'celery' package to enable async tasks."
            )


def make_celery(app_name: str = "tradingbotui") -> Celery:
    """Create a Celery instance configured from environment variables.

    Defaults to redis://localhost:6379/0, but gracefully falls back to an in-memory
    broker when CELERY_ALWAYS_EAGER=1 (default) or CELERY_USE_IN_MEMORY=1.
    """

    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend_url = os.getenv("CELERY_RESULT_BACKEND", broker_url)

    if os.getenv("CELERY_USE_IN_MEMORY", "0") == "1" or not _CELERY_AVAILABLE:
        broker_url = backend_url = "memory://"

    celery = Celery(app_name, broker=broker_url, backend=backend_url)
    conf = getattr(celery, "conf", None)
    if isinstance(conf, dict) or hasattr(conf, "update"):
        conf.update(
            task_always_eager=os.getenv("CELERY_ALWAYS_EAGER", "1") == "1",
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            timezone="UTC",
            enable_utc=True,
        )

    return celery


# Lazy singleton used across the package.
celery_app = make_celery()
