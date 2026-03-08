from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from openenv_app.openenv_wrapper import OpenEnvTrafficWrapper
from openenv_app.replay_runner import get_cached, run_and_cache
from openenv_app.schema import (
    HealthResponse,
    ReplayResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResponse,
)
from server.path_validators import validate_path_segment

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", "") or (_REPO_ROOT / "data" / "bundled"))
CHECKPOINT_PATH = Path(
    os.environ.get("CHECKPOINT_PATH", "")
    or (_REPO_ROOT / "artifacts" / "dqn_shared" / "best_validation.pt")
)

# ---------------------------------------------------------------------------
# Startup / lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the DQN checkpoint once at startup so replay requests are fast."""
    if CHECKPOINT_PATH.exists():
        from server.policy_runner import load_dqn_checkpoint
        load_dqn_checkpoint(CHECKPOINT_PATH)
    else:
        logger.warning("Checkpoint not found at %s — 'learned' policy will fail", CHECKPOINT_PATH)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DistrictFlow OpenEnv App",
    description="OpenEnv-style traffic environment for district-level LLM coordination.",
    version="0.1.0",
    lifespan=lifespan,
)

# Lazy-initialized: only constructed when /reset or /step is first called.
_wrapper: OpenEnvTrafficWrapper | None = None


def _get_wrapper() -> OpenEnvTrafficWrapper:
    global _wrapper
    if _wrapper is None:
        _wrapper = OpenEnvTrafficWrapper()
    return _wrapper


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(status="ok", message="DistrictFlow OpenEnv app is running.")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", message="healthy")


# ---------------------------------------------------------------------------
# Step / Reset / State
# ---------------------------------------------------------------------------


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    payload = _get_wrapper().reset(
        seed=request.seed,
        city_id=request.city_id,
        scenario_name=request.scenario_name,
    )
    return ResetResponse(observation=payload["observation"], info=payload.get("info", {}))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    payload = _get_wrapper().step(action=request.action)
    return StepResponse(
        observation=payload["observation"],
        reward=payload["reward"],
        done=payload["done"],
        truncated=payload.get("truncated", False),
        info=payload.get("info", {}),
    )


@app.get("/state", response_model=StateResponse)
def state():
    payload = _get_wrapper().state()
    return StateResponse(state=payload["state"])


# ---------------------------------------------------------------------------
# Replay  (on-demand simulation + in-memory cache)
# ---------------------------------------------------------------------------

_VALID_POLICIES = {"no_intervention", "fixed", "random", "learned"}


@app.get("/replay/{city_id}/{scenario_name}/{policy_name}", response_model=ReplayResponse)
def get_replay(city_id: str, scenario_name: str, policy_name: str) -> ReplayResponse:
    """Run a full simulation and return the CityFlow replay + metrics.

    Results are cached in memory so repeated calls are instant.
    """
    validate_path_segment(city_id, "city_id")
    validate_path_segment(scenario_name, "scenario_name")

    if policy_name not in _VALID_POLICIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown policy '{policy_name}'. Valid: {sorted(_VALID_POLICIES)}",
        )

    cached = get_cached(city_id, scenario_name, policy_name)
    if cached is None:
        try:
            cached = run_and_cache(
                city_id=city_id,
                scenario_name=scenario_name,
                policy_name=policy_name,
                generated_root=DATA_DIR,
            )
        except FileNotFoundError as exc:
            logger.error("Replay file missing after simulation: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Simulation completed but no replay file was produced.",
            ) from exc
        except Exception as exc:
            logger.error("Simulation failed for %s/%s/%s: %s", city_id, scenario_name, policy_name, exc)
            raise HTTPException(status_code=500, detail="Simulation failed.") from exc

    replay_text, roadnet_log, metrics = cached
    return ReplayResponse(
        city_id=city_id,
        scenario_name=scenario_name,
        policy_name=policy_name,
        replay_text=replay_text,
        roadnet_log=roadnet_log,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
def unhandled_exception_handler(request, exc):
    logger.error("Unhandled exception: %s: %s", type(exc).__name__, exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
