from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    seed: int | None = None
    city_id: str | None = None
    scenario_name: str | None = None


class StepRequest(BaseModel):
    action: dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    observation: dict[str, Any]
    info: dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    state: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    message: str
