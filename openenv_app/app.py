from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from openenv_app.openenv_wrapper import OpenEnvTrafficWrapper
from openenv_app.schema import (
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResponse,
)

app = FastAPI(
    title="DistrictFlow OpenEnv App",
    description="OpenEnv-style traffic environment for district-level LLM coordination.",
    version="0.1.0",
)

wrapper = OpenEnvTrafficWrapper()


@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(
        status="ok",
        message="DistrictFlow OpenEnv app is running.",
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        message="healthy",
    )


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    payload = wrapper.reset(
        seed=request.seed,
        city_id=request.city_id,
        scenario_name=request.scenario_name,
    )
    return ResetResponse(
        observation=payload["observation"],
        info=payload.get("info", {}),
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    payload = wrapper.step(action=request.action)
    return StepResponse(
        observation=payload["observation"],
        reward=payload["reward"],
        done=payload["done"],
        truncated=payload.get("truncated", False),
        info=payload.get("info", {}),
    )


@app.get("/state", response_model=StateResponse)
def state():
    payload = wrapper.state()
    return StateResponse(state=payload["state"])


@app.exception_handler(Exception)
def unhandled_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
        },
    )
