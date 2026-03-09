from __future__ import annotations

import json
from typing import Any

from openenv.core.env_server import Action, Observation, State
from pydantic import ConfigDict, Field, field_validator


class AgenticTrafficAction(Action):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        validate_default=True,
    )

    use_llm: bool = Field(
        default=False,
        description=(
            "When true, use the bundled district LLM adapter to generate district_actions "
            "for districts not explicitly provided."
        ),
    )
    district_actions: dict[str, Any] = Field(
        default="{}",
        description=(
            "JSON object keyed by district_id. Use {} for a no-op step, or provide "
            'entries like {"d_00":{"strategy":"hold","phase_bias":"NS","duration_steps":10}}.'
        ),
        json_schema_extra={
            "type": "string",
            "maxLength": 4000,
            "default": "{}",
        },
    )
    llm_max_new_tokens: int = Field(
        default=128,
        ge=16,
        le=512,
        description="Maximum new tokens to generate per district when use_llm=true.",
    )

    @field_validator("district_actions", mode="before")
    @classmethod
    def parse_district_actions(cls, value: Any) -> dict[str, Any]:
        if value is None or value == "":
            return {}
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("district_actions must decode to a JSON object.")
            return parsed
        if isinstance(value, dict):
            return value
        raise ValueError("district_actions must be a dict or JSON object string.")


class AgenticTrafficObservation(Observation):
    city_id: str | None = None
    scenario_name: str | None = None
    decision_step: int = 0
    sim_time: int = 0
    district_summaries: dict[str, Any] = Field(default_factory=dict)


class AgenticTrafficState(State):
    scenario: dict[str, Any] | None = None
    controller: dict[str, Any] = Field(default_factory=dict)
    district_decision_interval: int = 0
    district_summaries: dict[str, Any] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    last_info: dict[str, Any] = Field(default_factory=dict)
