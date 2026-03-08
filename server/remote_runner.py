"""HTTP client that delegates simulation runs to a remote OpenEnv API Space."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def run_policy_remote(
    city_id: str,
    scenario_name: str,
    policy_name: str,
    openenv_api_url: str,
    output_root: Path,
    timeout: float = 120.0,
):
    """Call Space 1's /replay endpoint and write results to output_root."""
    from server.policy_runner import RunResult

    url = f"{openenv_api_url.rstrip('/')}/replay/{city_id}/{scenario_name}/{policy_name}"
    logger.info("Remote replay request: %s", url)

    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url)
        resp.raise_for_status()

    payload: dict[str, Any] = resp.json()

    output_dir = output_root / city_id / scenario_name / policy_name
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_path = output_dir / "replay.txt"
    replay_path.write_text(payload["replay_text"], encoding="utf-8")

    roadnet_log_path = output_dir / "roadnetLogFile.json"
    if payload.get("roadnet_log"):
        roadnet_log_path.write_text(
            json.dumps(payload["roadnet_log"], indent=2), encoding="utf-8"
        )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload.get("metrics", {}), indent=2), encoding="utf-8"
    )

    return RunResult(
        city_id=city_id,
        scenario_name=scenario_name,
        policy_name=policy_name,
        replay_path=replay_path,
        roadnet_log_path=roadnet_log_path,
        metrics=payload.get("metrics", {}),
    )
