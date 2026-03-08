from __future__ import annotations

import os
from pathlib import Path

from models import (
    AgenticTrafficAction,
    AgenticTrafficObservation,
    AgenticTrafficState,
)
from openenv.core.env_server.interfaces import Environment
from openenv_app.openenv_wrapper import OpenEnvTrafficWrapper


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", "") or (REPO_ROOT / "data" / "generated"))
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", "") or (REPO_ROOT / "data" / "splits"))


class AgenticTrafficEnvironment(
    Environment[AgenticTrafficAction, AgenticTrafficObservation, AgenticTrafficState]
):
    """Minimal OpenEnv-compatible wrapper around the existing district controller stack."""

    def __init__(self) -> None:
        self.wrapper = OpenEnvTrafficWrapper(
            generated_root=DATA_DIR,
            splits_root=SPLITS_DIR,
        )
        self._state = AgenticTrafficState()

    def reset(self) -> AgenticTrafficObservation:
        payload = self.wrapper.reset(seed=None)
        self._sync_state()
        return AgenticTrafficObservation.model_validate(payload["observation"])

    def step(self, action: AgenticTrafficAction) -> AgenticTrafficObservation:
        payload = self.wrapper.step(action=action.model_dump())
        self._sync_state()
        observation = AgenticTrafficObservation.model_validate(payload["observation"])
        observation.done = bool(payload.get("done", False))
        observation.reward = float(payload.get("reward", 0.0))
        return observation

    @property
    def state(self) -> AgenticTrafficState:
        self._sync_state()
        return self._state

    def _sync_state(self) -> None:
        payload = self.wrapper.state()["state"]
        self._state = AgenticTrafficState.model_validate(payload)
