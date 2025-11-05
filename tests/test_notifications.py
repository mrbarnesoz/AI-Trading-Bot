import os
os.environ.setdefault("DUCKDB_PATH", ":memory:")

import json

from tradingbotui import tasks


def test_record_guardrail_violation_sends_slack(monkeypatch, tmp_path):
    log_path = tmp_path / "guardrail.jsonl"
    events = []

    monkeypatch.setattr(tasks, "GUARDRAIL_LOG", log_path, raising=False)
    monkeypatch.setattr(tasks, "SLACK_WEBHOOK_URL", "https://example.com/webhook", raising=False)
    monkeypatch.setattr(tasks, "_post_slack_message", lambda text, **_: events.append(text))

    event = {
        "timestamp": "2025-01-01T00:00:00Z",
        "stage": "live",
        "reason": "Risk exceeded",
        "config": "configs/example.yaml",
        "symbol": "XBTUSD",
    }

    tasks._record_guardrail_violation(event)

    assert log_path.exists()
    with log_path.open() as handle:
        stored = json.loads(handle.readline())
    assert stored["stage"] == "live"
    assert events and "Guardrail" in events[0]
