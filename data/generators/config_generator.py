"""CityFlow config generation for per-scenario simulation runs."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any


class ConfigGenerator:
    """Build CityFlow-compatible config.json payloads."""

    def generate(
        self,
        simulation_steps: int,
        interval: float,
        seed: int,
        save_replay: bool,
        roadnet_file: Path,
        flow_file: Path,
        scenario_dir: Path,
    ) -> dict[str, Any]:
        # CityFlow expects a small signed integer for seed.
        safe_seed = int(seed) & 0x7FFFFFFF
        # Use absolute paths so CityFlow can resolve files regardless of working dir.
        roadnet_path = roadnet_file.resolve()
        flow_path = flow_file.resolve()
        scenario_path = scenario_dir.resolve()
        base_dir = roadnet_path.parent
        roadnet_rel = os.path.relpath(roadnet_path, base_dir)
        flow_rel = os.path.relpath(flow_path, base_dir)
        flow_rel_dir = Path(flow_rel).parent
        roadnet_log_rel = str(flow_rel_dir / "roadnetLogFile.json")
        replay_log_rel = str(flow_rel_dir / "replay.txt")
        dir_str = str(base_dir)
        if not dir_str.endswith(os.sep):
            dir_str = dir_str + os.sep
        return {
            "interval": interval,
            "seed": safe_seed,
            "dir": dir_str,
            "roadnetFile": roadnet_rel,
            "flowFile": flow_rel,
            "rlTrafficLight": True,
            "laneChange": False,
            "saveReplay": save_replay,
            "roadnetLogFile": roadnet_log_rel,
            "replayLogFile": replay_log_rel,
            "step": simulation_steps,
        }
