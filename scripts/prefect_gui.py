"""Simple Dash GUI for triggering Prefect deployments with parameter controls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List

from dash import ALL, Dash, Input, Output, State, dcc, html

# Deployment configuration
DEPLOYMENTS: Dict[str, Dict] = {
    "bitmex-etl-flow/etl-bitmex": {
        "description": "Run the ETL pipeline for a given date span and symbol list.",
        "params": [
            {"name": "start", "label": "Start Date (YYYY-MM-DD)", "default": "2014-11-22"},
            {"name": "end", "label": "End Date (YYYY-MM-DD)", "default": "2014-11-22"},
            {"name": "symbols", "label": "Symbols (JSON array)", "default": '["XBTUSD"]'},
        ],
    },
    "bitmex-qc-flow/qc-bitmex": {
        "description": "Run QC checks against bronze and silver data.",
        "params": [],
    },
    "bitmex-daily-flow/daily-bitmex": {
        "description": "End-to-end DAG (download, pipeline, QC, report).",
        "params": [],
    },
}


app = Dash(__name__)
app.title = "Prefect Deployment Runner"


def build_param_inputs(selected: str) -> List[html.Div]:
    meta = DEPLOYMENTS[selected]["params"]
    children: List[html.Div] = []
    for param in meta:
        children.append(
            html.Div(
                [
                    html.Label(param["label"]),
                    dcc.Input(
                        id={"type": "param-input", "name": param["name"]},
                        value=param.get("default", ""),
                        type="text",
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "0.75rem"},
            )
        )
    if not children:
        children.append(html.Div("No parameters required for this deployment.", style={"fontStyle": "italic"}))
    return children


app.layout = html.Div(
    [
        html.H2("Prefect Deployment Runner"),
        html.Div(
            [
                html.Label("Select Deployment"),
                dcc.Dropdown(
                    id="deployment-dropdown",
                    options=[{"label": name, "value": name} for name in DEPLOYMENTS.keys()],
                    value="bitmex-etl-flow/etl-bitmex",
                    clearable=False,
                ),
                html.Div(id="deployment-description", style={"marginTop": "0.5rem", "fontStyle": "italic"}),
            ],
            style={"maxWidth": "600px"},
        ),
        html.Hr(),
        html.Div(id="param-container"),
        html.Button("Run Deployment", id="run-button", n_clicks=0, style={"marginTop": "1rem"}),
        dcc.Loading(html.Div(id="command-output", style={"whiteSpace": "pre-wrap", "marginTop": "1.5rem"})),
    ],
    style={"fontFamily": "sans-serif", "maxWidth": "700px", "margin": "2rem auto"},
)


@app.callback(
    Output("param-container", "children"),
    Output("deployment-description", "children"),
    Input("deployment-dropdown", "value"),
)
def update_inputs(selected: str):
    if selected is None:
        selected = "bitmex-etl-flow/etl-bitmex"
    inputs = build_param_inputs(selected)
    description = DEPLOYMENTS[selected]["description"]
    return inputs, description


@app.callback(
    Output("command-output", "children"),
    Input("run-button", "n_clicks"),
    State("deployment-dropdown", "value"),
    State({"type": "param-input", "name": ALL}, "id"),
    State({"type": "param-input", "name": ALL}, "value"),
    prevent_initial_call=True,
)
def run_deployment(n_clicks: int, deployment: str, ids, values):
    if n_clicks == 0:
        return ""

    params_meta = {item["name"]: item for item in DEPLOYMENTS[deployment]["params"]}
    params: Dict[str, str] = {}
    for id_obj, value in zip(ids, values):
        name = id_obj["name"]
        if isinstance(value, str):
            value = value.strip()
        if value is None or value == "":
            continue
        params[name] = value

    cmd = [sys.executable, "-m", "prefect", "--no-color", "deployment", "run", deployment]
    message_lines = [f"$ {' '.join(cmd)}"]

    if params:
        try:
            parsed = {k: json.loads(v) if k == "symbols" else v for k, v in params.items()}
        except json.JSONDecodeError as exc:
            return f"Invalid JSON in parameters: {exc}"
        params_json = json.dumps(parsed)
        cmd.extend(["--params", params_json])
        message_lines.append(f"Parameters: {params_json}")

    try:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PREFECT_API_URL", env.get("PREFECT_API_URL", "http://127.0.0.1:4200/api"))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        if result.stdout:
            message_lines.append(result.stdout)
        if result.stderr:
            message_lines.append(result.stderr)
        message_lines.append("Status: completed with exit code 0")
    except subprocess.CalledProcessError as exc:
        message_lines.append(f"Status: failed with exit code {exc.returncode}")
        if exc.stdout:
            message_lines.append(exc.stdout)
        if exc.stderr:
            message_lines.append(exc.stderr)

    return "\n".join(message_lines)


if __name__ == "__main__":
    app.run(debug=True)
