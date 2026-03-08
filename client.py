from __future__ import annotations

from typing import Any

import requests

from models import (
    AgenticTrafficAction,
    AgenticTrafficObservation,
    AgenticTrafficState,
)


class AgenticTrafficClient:
    """Thin HTTP client for the DistrictFlow OpenEnv server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, seed: int | None = None) -> AgenticTrafficObservation:
        response = requests.post(
            f"{self.base_url}/reset",
            json={"seed": seed},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return AgenticTrafficObservation.model_validate(payload["observation"])

    def step(self, action: AgenticTrafficAction) -> AgenticTrafficObservation:
        response = requests.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        observation = AgenticTrafficObservation.model_validate(payload["observation"])
        observation.done = bool(payload.get("done", False))
        observation.reward = float(payload.get("reward", 0.0))
        return observation

    def state(self) -> AgenticTrafficState:
        response = requests.get(f"{self.base_url}/state", timeout=60)
        response.raise_for_status()
        payload = response.json()
        return AgenticTrafficState.model_validate(payload["state"])

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=30)
        response.raise_for_status()
        return response.json()
