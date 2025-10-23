from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs"

app = FastAPI(
    title="AI Trading Bot Orchestration API",
    version="0.1.0",
    summary="Lightweight diagnostics for RunBot-managed services.",
)


def _collect_logs() -> List[str]:
    if not LOG_ROOT.exists():
        return []
    names = {p.name for p in LOG_ROOT.glob("*.log")}
    names.update(p.name for p in LOG_ROOT.glob("*.out.log"))
    names.update(p.name for p in LOG_ROOT.glob("*.err.log"))
    return sorted(names)


@app.get("/health", tags=["status"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/logs", response_model=List[str], tags=["logs"])
def list_logs() -> List[str]:
    return _collect_logs()


@app.get("/logs/{log_name}", tags=["logs"])
def read_log(log_name: str, tail: int = 200) -> dict[str, object]:
    path = LOG_ROOT / log_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Log not found")
    tail = max(1, min(tail, 2000))
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    snippet = lines[-tail:]
    return {"log": log_name, "tail": tail, "lines": snippet}
