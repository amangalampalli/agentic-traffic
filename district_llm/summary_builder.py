from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from district_llm.schema import CongestedIntersection, DistrictStateSummary
from env.utils import load_json


@dataclass
class _SummaryContext:
    previous_summaries: dict[str, DistrictStateSummary]
    previous_finished_vehicles: int


class DistrictStateSummaryBuilder:
    def __init__(self, top_k: int = 3):
        self.top_k = int(top_k)
        self._context = _SummaryContext(previous_summaries={}, previous_finished_vehicles=0)
        self._scenario_metadata: dict[str, Any] | None = None

    def reset(self) -> None:
        self._context = _SummaryContext(previous_summaries={}, previous_finished_vehicles=0)
        self._scenario_metadata = None

    def build_all(self, env, observation_batch: dict[str, Any]) -> dict[str, DistrictStateSummary]:
        if self._scenario_metadata is None:
            metadata_path = Path(env.scenario_dir) / "scenario_metadata.json"
            self._scenario_metadata = load_json(metadata_path) if metadata_path.exists() else {}

        lane_vehicle_count = env.adapter.get_lane_vehicle_count()
        finished_vehicles = int(env.adapter.get_finished_vehicle_count())
        district_summaries: dict[str, DistrictStateSummary] = {}

        for district_id in env.districts:
            district_summaries[district_id] = self._build_single(
                env=env,
                observation_batch=observation_batch,
                lane_vehicle_count=lane_vehicle_count,
                district_id=district_id,
                finished_vehicles=finished_vehicles,
            )

        self._context.previous_summaries = district_summaries
        self._context.previous_finished_vehicles = finished_vehicles
        return district_summaries

    def _build_single(
        self,
        env,
        observation_batch: dict[str, Any],
        lane_vehicle_count: dict[str, int],
        district_id: str,
        finished_vehicles: int,
    ) -> DistrictStateSummary:
        district = env.districts[district_id]
        scenario_metadata = self._scenario_metadata or {}
        intersection_ids = observation_batch["intersection_ids"]
        district_ids = observation_batch["district_ids"]
        incoming_counts = observation_batch["incoming_counts"]
        incoming_waiting = observation_batch["incoming_waiting"]
        current_phase = observation_batch["current_phase"]

        queue_totals: list[float] = []
        wait_totals: list[float] = []
        outgoing_loads: list[float] = []
        ns_queue = 0.0
        ew_queue = 0.0
        ns_wait = 0.0
        ew_wait = 0.0
        boundary_queue_total = 0.0
        boundary_wait_total = 0.0
        congestion_items: list[CongestedIntersection] = []

        for index, intersection_id in enumerate(intersection_ids):
            if district_ids[index] != district_id:
                continue

            queue_total = float(np.asarray(incoming_counts[index], dtype=np.float32).sum())
            wait_total = float(np.asarray(incoming_waiting[index], dtype=np.float32).sum())
            outgoing_load = self._compute_outgoing_load(
                env=env,
                lane_vehicle_count=lane_vehicle_count,
                intersection_id=intersection_id,
            )
            queue_totals.append(queue_total)
            wait_totals.append(wait_total)
            outgoing_loads.append(outgoing_load)

            midpoint = incoming_counts.shape[1] // 2
            ns_queue += float(np.asarray(incoming_counts[index][:midpoint], dtype=np.float32).sum())
            ew_queue += float(np.asarray(incoming_counts[index][midpoint:], dtype=np.float32).sum())
            ns_wait += float(np.asarray(incoming_waiting[index][:midpoint], dtype=np.float32).sum())
            ew_wait += float(np.asarray(incoming_waiting[index][midpoint:], dtype=np.float32).sum())

            intersection_config = env.intersections[intersection_id]
            if intersection_config.is_boundary:
                boundary_queue_total += queue_total
                boundary_wait_total += wait_total

            congestion_items.append(
                CongestedIntersection(
                    intersection_id=intersection_id,
                    queue_total=queue_total,
                    wait_total=wait_total,
                    outgoing_load=outgoing_load,
                    current_phase=int(current_phase[index]),
                    is_boundary=bool(intersection_config.is_boundary),
                )
            )

        queue_array = np.asarray(queue_totals or [0.0], dtype=np.float32)
        wait_array = np.asarray(wait_totals or [0.0], dtype=np.float32)
        outgoing_array = np.asarray(outgoing_loads or [0.0], dtype=np.float32)

        previous_summary = self._context.previous_summaries.get(district_id)
        recent_throughput = float(
            finished_vehicles - self._context.previous_finished_vehicles
            if self._context.previous_finished_vehicles
            else 0.0
        )
        queue_change = 0.0 if previous_summary is None else float(queue_array.sum() - previous_summary.total_queue)
        wait_change = 0.0 if previous_summary is None else float(wait_array.sum() - previous_summary.total_wait)
        throughput_change = (
            0.0
            if previous_summary is None
            else recent_throughput - previous_summary.recent_throughput
        )

        directional_ns = ns_queue + 1.5 * ns_wait
        directional_ew = ew_queue + 1.5 * ew_wait
        if directional_ns > directional_ew * 1.1:
            dominant_flow = "NS"
        elif directional_ew > directional_ns * 1.1:
            dominant_flow = "EW"
        else:
            dominant_flow = "BALANCED"

        boundary_share = boundary_queue_total / max(1.0, float(queue_array.sum()))
        spillback_risk = bool(
            outgoing_array.max() >= max(8.0, queue_array.max() * 0.5)
            or (boundary_share >= 0.6 and queue_change >= 0.0)
        )

        top_intersections = sorted(
            congestion_items,
            key=lambda item: (item.queue_total + 1.5 * item.wait_total + 0.5 * item.outgoing_load),
            reverse=True,
        )[: self.top_k]

        overload_flag = bool(
            scenario_metadata.get("overload_district") == district_id
            or (scenario_metadata.get("name") == "district_overload" and queue_array.sum() >= 25.0)
        )
        event_flag = bool(scenario_metadata.get("event_district") == district_id)
        incident_flag = bool(
            scenario_metadata.get("name") in {"accident", "construction"}
            or bool(scenario_metadata.get("blocked_roads"))
        )
        construction_flag = bool(scenario_metadata.get("name") == "construction")

        return DistrictStateSummary(
            city_id=env.city_id,
            district_id=district_id,
            district_type=district.district_type,
            scenario_name=env.scenario_name,
            scenario_type=str(scenario_metadata.get("intensity", env.scenario_name)),
            decision_step=int(observation_batch["decision_step"]),
            sim_time=int(observation_batch["sim_time"]),
            intersection_count=int(len(district.intersection_ids)),
            avg_queue=float(queue_array.mean()),
            max_queue=float(queue_array.max()),
            total_queue=float(queue_array.sum()),
            avg_wait=float(wait_array.mean()),
            max_wait=float(wait_array.max()),
            total_wait=float(wait_array.sum()),
            avg_outgoing_load=float(outgoing_array.mean()),
            max_outgoing_load=float(outgoing_array.max()),
            total_outgoing_load=float(outgoing_array.sum()),
            recent_throughput=recent_throughput,
            queue_change=queue_change,
            wait_change=wait_change,
            throughput_change=throughput_change,
            ns_queue=ns_queue,
            ew_queue=ew_queue,
            ns_wait=ns_wait,
            ew_wait=ew_wait,
            dominant_flow=dominant_flow,
            boundary_queue_total=boundary_queue_total,
            boundary_wait_total=boundary_wait_total,
            spillback_risk=spillback_risk,
            incident_flag=incident_flag,
            construction_flag=construction_flag,
            overload_flag=overload_flag,
            event_flag=event_flag,
            top_congested_intersections=top_intersections,
        ).validate()

    @staticmethod
    def _compute_outgoing_load(env, lane_vehicle_count: dict[str, int], intersection_id: str) -> float:
        intersection_config = env.intersections[intersection_id]
        if not intersection_config.outgoing_lanes:
            return 0.0
        return float(
            sum(float(lane_vehicle_count.get(lane_id, 0)) for lane_id in intersection_config.outgoing_lanes)
        )
