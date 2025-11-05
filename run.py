"""Entry point for the Trading Bot web UI."""

from tradingbotui import create_app

app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
