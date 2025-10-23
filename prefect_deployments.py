"""Prefect deployment registration script compatible with Prefect 3."""

from __future__ import annotations

from datetime import timedelta

from prefect.client.schemas.schedules import IntervalSchedule
from prefect.deployments.runner import EntrypointType

from orchestration.prefect_flows import bitmex_daily_flow, bitmex_etl_flow, bitmex_qc_flow

WORK_POOL = "bitmex"
CODE_PATH = "."


def register_deployments() -> None:
    bitmex_daily_flow.deploy(
        name="daily-bitmex",
        work_pool_name=WORK_POOL,
        schedule=IntervalSchedule(interval=timedelta(days=1)),
        build=False,
        push=False,
        entrypoint_type=EntrypointType.MODULE_PATH,
        ignore_warnings=True,
    )
    bitmex_etl_flow.deploy(
        name="etl-bitmex",
        work_pool_name=WORK_POOL,
        build=False,
        push=False,
        entrypoint_type=EntrypointType.MODULE_PATH,
        ignore_warnings=True,
    )
    bitmex_qc_flow.deploy(
        name="qc-bitmex",
        work_pool_name=WORK_POOL,
        build=False,
        push=False,
        entrypoint_type=EntrypointType.MODULE_PATH,
        ignore_warnings=True,
    )
    print("Prefect deployments registered (daily, etl, qc) under work pool 'bitmex'.")


if __name__ == "__main__":
    register_deployments()
