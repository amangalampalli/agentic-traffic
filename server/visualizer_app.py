"""FastAPI visualizer server for CityFlow multi-policy comparison dashboard.

Deployment modes
----------------
Local (no OPENENV_API_URL set):
    Runs CityFlow simulations in-process via ``server.policy_runner``.

HF Space 2 (OPENENV_API_URL set to Space 1's URL):
    Delegates all simulation runs to the remote OpenEnv API via
    ``server.remote_runner``.  No CityFlow or torch needed locally.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from joblib import Parallel, delayed
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------

OPENENV_API_URL: str | None = os.environ.get("OPENENV_API_URL") or None

GENERATED_ROOT = Path(
    os.environ.get("DATA_DIR", "") or (REPO_ROOT / "data" / "generated")
)
REPLAY_OUTPUT_ROOT = Path(
    os.environ.get("REPLAY_ROOT", "") or (REPO_ROOT / "results" / "replays")
)
CHECKPOINT_PATH = Path(
    os.environ.get("CHECKPOINT_PATH", "")
    or (REPO_ROOT / "artifacts" / "dqn_shared" / "best_validation.pt")
)
FRONTEND_DIR = REPO_ROOT / "third_party" / "CityFlow" / "frontend"

# ---------------------------------------------------------------------------
# Runner selection: local (policy_runner) vs. remote (remote_runner)
# ---------------------------------------------------------------------------

from server.path_validators import validate_path_segment
from server.policy_runner import ALL_POLICIES, RunResult

if OPENENV_API_URL:
    logger.info("Remote mode — OpenEnv API at %s", OPENENV_API_URL)
    from server.remote_runner import run_policy_remote as _run_remote

    def _run_policy(city_id: str, scenario_name: str, policy_name: str) -> RunResult:
        return _run_remote(
            city_id=city_id,
            scenario_name=scenario_name,
            policy_name=policy_name,
            openenv_api_url=OPENENV_API_URL,  # type: ignore[arg-type]
            output_root=REPLAY_OUTPUT_ROOT,
        )

else:
    logger.info("Local mode — running CityFlow in-process")
    from server.policy_runner import run_policy_for_city as _run_local

    def _run_policy(city_id: str, scenario_name: str, policy_name: str) -> RunResult:
        return _run_local(
            city_id=city_id,
            scenario_name=scenario_name,
            policy_name=policy_name,
            generated_root=GENERATED_ROOT,
            output_root=REPLAY_OUTPUT_ROOT,
        )


from server.roadnet_matcher import (
    list_all_cities,
    list_scenarios_for_city,
    match_city_by_roadnet,
)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    REPLAY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if OPENENV_API_URL:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{OPENENV_API_URL.rstrip('/')}/health")
                resp.raise_for_status()
            logger.info("OpenEnv API health check passed: %s", OPENENV_API_URL)
        except Exception as exc:
            logger.warning(
                "OpenEnv API at %s did not respond to /health: %s. "
                "Simulation requests will fail until it is reachable.",
                OPENENV_API_URL,
                exc,
            )
    else:
        from server.policy_runner import load_dqn_checkpoint
        if CHECKPOINT_PATH.exists():
            load_dqn_checkpoint(CHECKPOINT_PATH)
        else:
            logger.warning("Checkpoint not found at %s — 'learned' policy will fail", CHECKPOINT_PATH)

    yield


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Traffic Visualizer",
    description="Multi-policy CityFlow replay comparison dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunSimulationsRequest(BaseModel):
    city_id: str
    scenario_name: str
    policies: list[str] = list(ALL_POLICIES)
    force: bool = False


class PolicyMetrics(BaseModel):
    policy_name: str
    metrics: dict[str, Any]
    replay_available: bool
    roadnet_log_available: bool


class RunSimulationsResponse(BaseModel):
    city_id: str
    scenario_name: str
    results: list[PolicyMetrics]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"status": "Traffic Visualizer API running"})


@app.post("/upload-roadnet")
async def upload_roadnet(file: UploadFile) -> dict:
    raw = await file.read()
    try:
        roadnet_data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    city_id = match_city_by_roadnet(roadnet_data, GENERATED_ROOT)
    if city_id is None:
        return {
            "matched": False,
            "city_id": None,
            "scenarios": [],
            "all_cities": list_all_cities(GENERATED_ROOT),
        }

    scenarios = list_scenarios_for_city(city_id, GENERATED_ROOT)
    return {"matched": True, "city_id": city_id, "scenarios": scenarios, "all_cities": []}


@app.get("/cities")
def get_cities() -> dict:
    return {"cities": list_all_cities(GENERATED_ROOT)}


@app.get("/cities/{city_id}/scenarios")
def get_scenarios(city_id: str) -> dict:
    validate_path_segment(city_id, "city_id")
    scenarios = list_scenarios_for_city(city_id, GENERATED_ROOT)
    if not scenarios:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city_id}' not found or has no scenarios.",
        )
    return {"city_id": city_id, "scenarios": scenarios}


@app.post("/run-simulations", response_model=RunSimulationsResponse)
def run_simulations(request: RunSimulationsRequest) -> RunSimulationsResponse:
    validate_path_segment(request.city_id, "city_id")
    validate_path_segment(request.scenario_name, "scenario_name")

    valid_policies = set(ALL_POLICIES)
    bad = [p for p in request.policies if p not in valid_policies]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown policies: {bad}. Valid: {list(ALL_POLICIES)}",
        )

    def _run_one(policy_name: str) -> PolicyMetrics:
        output_dir = REPLAY_OUTPUT_ROOT / request.city_id / request.scenario_name / policy_name
        replay_path = output_dir / "replay.txt"
        roadnet_path = output_dir / "roadnetLogFile.json"
        metrics_path = output_dir / "metrics.json"

        if not request.force and replay_path.exists() and metrics_path.exists():
            return PolicyMetrics(
                policy_name=policy_name,
                metrics=json.loads(metrics_path.read_text(encoding="utf-8")),
                replay_available=True,
                roadnet_log_available=roadnet_path.exists(),
            )

        try:
            result: RunResult = _run_policy(
                city_id=request.city_id,
                scenario_name=request.scenario_name,
                policy_name=policy_name,
            )
            return PolicyMetrics(
                policy_name=policy_name,
                metrics=result.metrics,
                replay_available=result.replay_path.exists(),
                roadnet_log_available=result.roadnet_log_path.exists(),
            )
        except Exception as exc:
            logger.error("Policy run failed for %s/%s/%s: %s", request.city_id, request.scenario_name, policy_name, exc)
            return PolicyMetrics(
                policy_name=policy_name,
                metrics={"error": "Simulation failed. Check server logs."},
                replay_available=False,
                roadnet_log_available=False,
            )

    n_jobs = min(len(request.policies), 4)
    results: list[PolicyMetrics] = Parallel(
        n_jobs=n_jobs, prefer="threads"
    )(delayed(_run_one)(p) for p in request.policies)

    return RunSimulationsResponse(
        city_id=request.city_id,
        scenario_name=request.scenario_name,
        results=results,
    )


@app.get("/replay/{city_id}/{scenario_name}/{policy_name}", response_model=None)
def get_replay(
    city_id: str,
    scenario_name: str,
    policy_name: str,
    max_steps: int = 0,
) -> PlainTextResponse | FileResponse:
    validate_path_segment(city_id, "city_id")
    validate_path_segment(scenario_name, "scenario_name")
    validate_path_segment(policy_name, "policy_name")

    replay_path = REPLAY_OUTPUT_ROOT / city_id / scenario_name / policy_name / "replay.txt"
    if not replay_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Replay not found for {city_id}/{scenario_name}/{policy_name}. Run /run-simulations first.",
        )
    if max_steps > 0:
        lines: list[str] = []
        with open(replay_path, encoding="utf-8") as fh:
            for raw in fh:
                if raw.strip():
                    lines.append(raw.rstrip("\n"))
                    if len(lines) >= max_steps:
                        break
        return PlainTextResponse("\n".join(lines))
    return FileResponse(str(replay_path), media_type="text/plain")


@app.get("/roadnet-log/{city_id}/{scenario_name}/{policy_name}")
def get_roadnet_log(city_id: str, scenario_name: str, policy_name: str) -> JSONResponse:
    validate_path_segment(city_id, "city_id")
    validate_path_segment(scenario_name, "scenario_name")
    validate_path_segment(policy_name, "policy_name")

    path = REPLAY_OUTPUT_ROOT / city_id / scenario_name / policy_name / "roadnetLogFile.json"
    if not path.exists():
        for p in ALL_POLICIES:
            fallback = REPLAY_OUTPUT_ROOT / city_id / scenario_name / p / "roadnetLogFile.json"
            if fallback.exists():
                path = fallback
                break
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Roadnet log not found for {city_id}/{scenario_name}.",
        )
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/metrics/{city_id}/{scenario_name}")
def get_metrics(city_id: str, scenario_name: str) -> dict:
    validate_path_segment(city_id, "city_id")
    validate_path_segment(scenario_name, "scenario_name")

    base = REPLAY_OUTPUT_ROOT / city_id / scenario_name
    if not base.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No simulation results found for {city_id}/{scenario_name}.",
        )

    metrics: dict[str, Any] = {}
    for policy_dir in sorted(base.iterdir()):
        if not policy_dir.is_dir():
            continue
        metrics_path = policy_dir / "metrics.json"
        replay_path = policy_dir / "replay.txt"
        if metrics_path.exists():
            metrics[policy_dir.name] = json.loads(metrics_path.read_text(encoding="utf-8"))
        elif replay_path.exists():
            metrics[policy_dir.name] = {"replay_available": True}

    return {"city_id": city_id, "scenario_name": scenario_name, "metrics": metrics}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.visualizer_app:app", host="0.0.0.0", port=8080, reload=False)
