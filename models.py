from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgenticTrafficAction(BaseModel):
    district_actions: dict[str, Any] = Field(default_factory=dict)


class AgenticTrafficObservation(BaseModel):
    city_id: str | None = None
    scenario_name: str | None = None
    decision_step: int = 0
    sim_time: int = 0
    district_summaries: dict[str, Any] = Field(default_factory=dict)
    done: bool = False
    reward: float = 0.0


class AgenticTrafficState(BaseModel):
    scenario: dict[str, Any] | None = None
    controller: dict[str, Any] = Field(default_factory=dict)
    district_decision_interval: int = 0
    district_summaries: dict[str, Any] = Field(default_factory=dict)
    last_info: dict[str, Any] = Field(default_factory=dict)
