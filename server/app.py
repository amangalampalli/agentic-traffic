from __future__ import annotations

from openenv.core.env_server import create_app

from models import AgenticTrafficAction, AgenticTrafficObservation
from server.environment import AgenticTrafficEnvironment


app = create_app(
    AgenticTrafficEnvironment,
    AgenticTrafficAction,
    AgenticTrafficObservation,
)
