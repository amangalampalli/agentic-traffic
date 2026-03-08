"""FastAPI visualizer server for CityFlow multi-policy comparison dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from joblib import Parallel, delayed
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server.policy_runner import ALL_POLICIES, RunResult, load_dqn_checkpoint, run_policy_for_city
from server.roadnet_matcher import (
    list_all_cities,
    list_scenarios_for_city,
    match_city_by_roadnet,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GENERATED_ROOT = REPO_ROOT / "data" / "generated"
REPLAY_OUTPUT_ROOT = REPO_ROOT / "results" / "replays"
CHECKPOINT_PATH = REPO_ROOT / "artifacts" / "dqn_shared" / "best_validation.pt"
FRONTEND_DIR = REPO_ROOT / "third_party" / "CityFlow" / "frontend"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Traffic Visualizer",
    description="Multi-policy CityFlow replay comparison dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend files at root.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.on_event("startup")
def startup() -> None:
    if CHECKPOINT_PATH.exists():
        load_dqn_checkpoint(CHECKPOINT_PATH)
    else:
        print(f"[server] WARNING: checkpoint not found at {CHECKPOINT_PATH}")
    REPLAY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunSimulationsRequest(BaseModel):
    city_id: str
    scenario_name: str
    policies: list[str] = list(ALL_POLICIES)
    force: bool = False  # bypass cache and re-run even if results exist


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
    """Accept a roadnet.json upload and return the matched city_id + scenarios."""
    raw = await file.read()
    try:
        roadnet_data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    city_id = match_city_by_roadnet(roadnet_data, GENERATED_ROOT)
    if city_id is None:
        # Return all cities so the user can pick manually.
        return {
            "matched": False,
            "city_id": None,
            "scenarios": [],
            "all_cities": list_all_cities(GENERATED_ROOT),
        }

    scenarios = list_scenarios_for_city(city_id, GENERATED_ROOT)
    return {
        "matched": True,
        "city_id": city_id,
        "scenarios": scenarios,
        "all_cities": [],
    }


@app.get("/cities")
def get_cities() -> dict:
    cities = list_all_cities(GENERATED_ROOT)
    return {"cities": cities}


@app.get("/cities/{city_id}/scenarios")
def get_scenarios(city_id: str) -> dict:
    scenarios = list_scenarios_for_city(city_id, GENERATED_ROOT)
    if not scenarios:
        raise HTTPException(status_code=404, detail=f"City '{city_id}' not found or has no scenarios.")
    return {"city_id": city_id, "scenarios": scenarios}


@app.post("/run-simulations", response_model=RunSimulationsResponse)
def run_simulations(request: RunSimulationsRequest) -> RunSimulationsResponse:
    """Run requested policies on the given city/scenario and generate replay files."""
    valid_policies = set(ALL_POLICIES)
    bad = [p for p in request.policies if p not in valid_policies]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown policies: {bad}. Valid: {list(ALL_POLICIES)}",
        )

    def _run_one(policy_name: str) -> PolicyMetrics:
        output_dir     = REPLAY_OUTPUT_ROOT / request.city_id / request.scenario_name / policy_name
        replay_path    = output_dir / "replay.txt"
        roadnet_path   = output_dir / "roadnetLogFile.json"
        metrics_path   = output_dir / "metrics.json"

        # Serve from cache when all files exist and force-rerun is not requested.
        if not request.force and replay_path.exists() and metrics_path.exists():
            print(f"[server] cache hit: {request.city_id}/{request.scenario_name}/{policy_name}")
            return PolicyMetrics(
                policy_name=policy_name,
                metrics=json.loads(metrics_path.read_text()),
                replay_available=True,
                roadnet_log_available=roadnet_path.exists(),
            )

        try:
            result: RunResult = run_policy_for_city(
                city_id=request.city_id,
                scenario_name=request.scenario_name,
                policy_name=policy_name,
                generated_root=GENERATED_ROOT,
                output_root=REPLAY_OUTPUT_ROOT,
            )
            return PolicyMetrics(
                policy_name=policy_name,
                metrics=result.metrics,
                replay_available=result.replay_path.exists(),
                roadnet_log_available=result.roadnet_log_path.exists(),
            )
        except Exception as exc:
            return PolicyMetrics(
                policy_name=policy_name,
                metrics={"error": str(exc)},
                replay_available=False,
                roadnet_log_available=False,
            )

    results: list[PolicyMetrics] = Parallel(
        n_jobs=-1, prefer="threads"
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
    """Return the replay.txt for a given city/scenario/policy.

    Pass ``?max_steps=N`` to limit the response to the first N non-empty lines
    (steps), keeping browser memory usage bounded for large cities.
    """
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
    """Return the roadnetLogFile.json generated by CityFlow for a given run."""
    path = REPLAY_OUTPUT_ROOT / city_id / scenario_name / policy_name / "roadnetLogFile.json"
    if not path.exists():
        # Fall back: any policy's roadnetLogFile works (static network is the same)
        for p in ALL_POLICIES:
            fallback = REPLAY_OUTPUT_ROOT / city_id / scenario_name / p / "roadnetLogFile.json"
            if fallback.exists():
                path = fallback
                break
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Roadnet log not found for {city_id}/{scenario_name}. Run /run-simulations first.",
        )
    return JSONResponse(json.loads(path.read_text()))


@app.get("/metrics/{city_id}/{scenario_name}")
def get_metrics(city_id: str, scenario_name: str) -> dict:
    """Return cached metrics for all policies that have been run on this city/scenario."""
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
            metrics[policy_dir.name] = json.loads(metrics_path.read_text())
        elif replay_path.exists():
            metrics[policy_dir.name] = {"replay_available": True}

    return {"city_id": city_id, "scenario_name": scenario_name, "metrics": metrics}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.visualizer_app:app", host="0.0.0.0", port=8080, reload=False)
