from __future__ import annotations

from pathlib import Path
from typing import Any


class CityFlowAdapter:
    def __init__(self, config_path: str | Path, thread_num: int = 1):
        self.config_path = str(config_path)
        self.thread_num = int(thread_num)
        self.engine = None
        self._phase_cache: dict[str, int] = {}
        self._active_vehicle_ids: set[str] = set()
        self._finished_vehicle_ids: set[str] = set()

    def reset(self) -> None:
        try:
            import cityflow
        except ImportError as exc:
            raise RuntimeError(
                "CityFlow is not installed. Install the CityFlow Python bindings "
                "before running smoke tests, training, or evaluation."
            ) from exc

        self.engine = cityflow.Engine(self.config_path, thread_num=self.thread_num)
        self._phase_cache.clear()
        self._active_vehicle_ids = self._fetch_active_vehicle_ids()
        self._finished_vehicle_ids.clear()

    def step(self) -> None:
        self._require_engine()
        self.engine.next_step()
        current_vehicle_ids = self._fetch_active_vehicle_ids()
        self._finished_vehicle_ids.update(self._active_vehicle_ids - current_vehicle_ids)
        self._active_vehicle_ids = current_vehicle_ids

    def set_tl_phase(self, intersection_id: str, phase: int) -> None:
        self._require_engine()
        self.engine.set_tl_phase(intersection_id, int(phase))
        self._phase_cache[intersection_id] = int(phase)

    def get_tl_phase(self, intersection_id: str) -> int:
        self._require_engine()
        if hasattr(self.engine, "get_tl_phase"):
            phase = int(self.engine.get_tl_phase(intersection_id))
            self._phase_cache[intersection_id] = phase
            return phase
        return int(self._phase_cache.get(intersection_id, 0))

    def get_lane_vehicle_count(self) -> dict[str, int]:
        self._require_engine()
        return {
            lane_id: int(count)
            for lane_id, count in self.engine.get_lane_vehicle_count().items()
        }

    def get_lane_waiting_vehicle_count(self) -> dict[str, int]:
        self._require_engine()
        return {
            lane_id: int(count)
            for lane_id, count in self.engine.get_lane_waiting_vehicle_count().items()
        }

    def get_current_time(self) -> int:
        self._require_engine()
        return int(self.engine.get_current_time())

    def get_vehicle_count(self) -> int:
        self._require_engine()
        if hasattr(self.engine, "get_vehicle_count"):
            return int(self.engine.get_vehicle_count())
        return len(self._active_vehicle_ids)

    def get_average_travel_time(self) -> float | None:
        self._require_engine()
        if hasattr(self.engine, "get_average_travel_time"):
            return float(self.engine.get_average_travel_time())
        return None

    def get_finished_vehicle_count(self) -> int:
        self._require_engine()
        if hasattr(self.engine, "get_finished_vehicle_count"):
            return int(self.engine.get_finished_vehicle_count())
        return len(self._finished_vehicle_ids)

    def get_active_vehicle_ids(self) -> set[str]:
        return set(self._active_vehicle_ids)

    def _fetch_active_vehicle_ids(self) -> set[str]:
        if self.engine is None or not hasattr(self.engine, "get_vehicles"):
            return set()

        vehicles = self.engine.get_vehicles()
        if isinstance(vehicles, dict):
            return set(vehicles.keys())
        return set(vehicles)

    def _require_engine(self) -> None:
        if self.engine is None:
            raise RuntimeError(
                "CityFlow engine has not been initialized. Call reset() before use."
            )
