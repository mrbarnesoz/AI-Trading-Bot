"""Flask application factory for the trading bot UI."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, request

from tradingbotui import routes
from ingestion.tradingview_webhook import blueprint as tradingview_blueprint


def _ensure_app_bundle() -> None:
    """Compile the React bundle if the source changed since the last build."""
    source_path = Path(__file__).resolve().parent / "static" / "js" / "app.jsx"
    bundle_path = Path(__file__).resolve().parent / "static" / "js" / "app.bundle.js"
    if not source_path.exists():
        return
    needs_build = not bundle_path.exists() or bundle_path.stat().st_mtime < source_path.stat().st_mtime
    if not needs_build:
        return
    npx_path = shutil.which("npx")
    if not npx_path:
        print("[tradingbotui] npx not found; skipping app bundle refresh.", flush=True)
        return
    cmd = [
        npx_path,
        "esbuild",
        str(source_path),
        "--loader:.jsx=jsx",
        f"--outfile={bundle_path}",
        "--format=iife",
        "--minify",
    ]
    try:
        print("[tradingbotui] Refreshing React bundle via esbuild...", flush=True)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
    except FileNotFoundError:
        print("[tradingbotui] esbuild not available; skipping bundle refresh.", flush=True)
    except subprocess.CalledProcessError as exc:
        print(f"[tradingbotui] esbuild failed ({exc.returncode}): {exc.stderr}", flush=True)


_ensure_app_bundle()


def create_app() -> Flask:
    """Create and configure the Flask application."""

    base_dir = os.getcwd()
    static_folder = os.path.join(os.path.dirname(__file__), "static")
    template_folder = os.path.join(os.path.dirname(__file__), "templates")

    app = Flask(
        __name__,
        static_folder=static_folder,
        template_folder=template_folder,
    )
    app.config["SECRET_KEY"] = os.environ.get("TRADINGBOT_UI_SECRET", "CHANGE_ME")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    # Configure logging directory
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "tradingbot_ui.log")

    if not any(isinstance(h, RotatingFileHandler) for h in app.logger.handlers):
        file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

    # Mirror logs to stdout so external process monitors can surface failures.
    if not any(isinstance(h, logging.StreamHandler) for h in app.logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info("TradingBot UI starting up")

    @app.before_request
    def _log_request() -> None:
        app.logger.info("HTTP %s %s", request.method, request.path)

    # Ensure template changes are reflected without manual cache clears.
    app.jinja_env.auto_reload = True

    # Register blueprints
    app.register_blueprint(routes.main_bp)
    app.register_blueprint(routes.api_bp)
    app.register_blueprint(tradingview_blueprint, url_prefix="/api/feeds")

    @app.before_request
    def _log_request() -> None:
        app.logger.info("HTTP %s %s", request.method, request.path)
        try:
            print(f"[REQUEST] {request.method} {request.path}", flush=True)
        except Exception:
            pass

    @app.after_request
    def _log_response(response):
        app.logger.info("HTTP %s %s -> %s", request.method, request.path, response.status_code)
        try:
            print(f"[RESPONSE] {request.method} {request.path} -> {response.status_code}", flush=True)
        except Exception:
            pass
        return response

    return app
