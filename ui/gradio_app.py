from __future__ import annotations

from pathlib import Path
from typing import List

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs"


def available_logs() -> List[str]:
    if not LOG_ROOT.exists():
        return []
    names = {p.name for p in LOG_ROOT.glob("*.log")}
    names.update(p.name for p in LOG_ROOT.glob("*.out.log"))
    names.update(p.name for p in LOG_ROOT.glob("*.err.log"))
    return sorted(names)


def read_log(name: str, tail: int = 200) -> str:
    if not name:
        return "Select a log to view its contents."
    path = LOG_ROOT / name
    if not path.exists():
        return f"Log {name} not found. Refresh the list if it was rotated."
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Unable to read {name}: {exc}"
    tail = max(1, min(int(tail or 200), 2000))
    snippet = content[-tail:]
    return "\n".join(snippet) if snippet else "(no log output yet)"


with gr.Blocks(title="AI Trading Bot — Logs") as demo:
    gr.Markdown("# AI Trading Bot — Log Viewer")
    gr.Markdown(
        "Use the dropdown to inspect the latest output from RunBot managed services. "
        "Restart RunBot if you add new log files and need them to appear here."
    )

    logs_state = gr.State(available_logs())

    with gr.Row():
        dropdown = gr.Dropdown(
            label="Available logs",
            choices=logs_state.value,
            allow_custom_value=False,
            value=logs_state.value[0] if logs_state.value else None,
        )
        tail_slider = gr.Slider(
            label="Lines",
            value=200,
            minimum=50,
            maximum=1000,
            step=50,
        )

    output = gr.Textbox(label="Log output", lines=35)
    dropdown.change(read_log, inputs=[dropdown, tail_slider], outputs=output, show_progress=False)
    tail_slider.change(read_log, inputs=[dropdown, tail_slider], outputs=output, show_progress=False)

    refresh_button = gr.Button("Refresh log list")

    def refresh_logs() -> gr.Dropdown:
        new_choices = available_logs()
        return gr.Dropdown(
            choices=new_choices,
            value=new_choices[0] if new_choices else None,
        )

    refresh_button.click(refresh_logs, outputs=dropdown)

    gr.Markdown(
        """
### Handy CLI commands
```
prefect deployment run 'bitmex-qc-flow/qc-bitmex'
prefect deployment run 'bitmex-daily-flow/daily-bitmex'
prefect deployment run 'bitmex-etl-flow/etl-bitmex' --params-file etl-params.json
```
Use a terminal in the activated RunBot environment to execute these workflows.
"""
    )


if __name__ == "__main__":
    demo.queue().launch()
