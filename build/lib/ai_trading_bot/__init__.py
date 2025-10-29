"""Top-level package for the AI Trading Bot project."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ai-trading-bot")
except PackageNotFoundError:  # pragma: no cover - during local runs without install
    __version__ = "0.1.0"

__all__ = ["__version__"]
