from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from district_llm.schema import CandidateIntersection, CongestedIntersection, DistrictStateSummary, candidate_priority_score
from env.utils import load_json


@dataclass
class _SummaryContext:
    previous_summaries: dict[str, DistrictStateSummary]
    previous_finished_vehicles: int


class DistrictStateSummaryBuilder:
    def __init__(self, top_k: int = 3, candidate_limit: int = 6):
        self.top_k = int(top_k)
        self.candidate_limit = int(candidate_limit)
        self._context = _SummaryContext(previous_summaries={}, previous_finished_vehicles=0)
        self._scenario_metadata: dict[str, Any] | None = None
        self._road_endpoints: dict[str, tuple[str, str]] | None = None
        self._incident_intersections: set[str] = set()

    def reset(self) -> None:
        self._context = _SummaryContext(previous_summaries={}, previous_finished_vehicles=0)
        self._scenario_metadata = None
        self._road_endpoints = None
        self._incident_intersections = set()

    def build_all(self, env, observation_batch: dict[str, Any]) -> dict[str, DistrictStateSummary]:
        if self._scenario_metadata is None:
            metadata_path = Path(env.scenario_dir) / "scenario_metadata.json"
            self._scenario_metadata = load_json(metadata_path) if metadata_path.exists() else {}
            self._road_endpoints = self._load_road_endpoints(Path(env.roadnet_path))
            self._incident_intersections = self._derive_incident_intersections()

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
        candidate_seed_items: list[dict[str, Any]] = []

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
            ns_queue_local = float(np.asarray(incoming_counts[index][:midpoint], dtype=np.float32).sum())
            ew_queue_local = float(np.asarray(incoming_counts[index][midpoint:], dtype=np.float32).sum())
            ns_wait_local = float(np.asarray(incoming_waiting[index][:midpoint], dtype=np.float32).sum())
            ew_wait_local = float(np.asarray(incoming_waiting[index][midpoint:], dtype=np.float32).sum())
            ns_queue += ns_queue_local
            ew_queue += ew_queue_local
            ns_wait += ns_wait_local
            ew_wait += ew_wait_local

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
            candidate_seed_items.append(
                {
                    "intersection_id": intersection_id,
                    "queue_total": queue_total,
                    "wait_total": wait_total,
                    "outgoing_load": outgoing_load,
                    "current_phase": int(current_phase[index]),
                    "is_boundary": bool(intersection_config.is_boundary),
                    "spillback_risk": bool(
                        outgoing_load >= max(6.0, queue_total * 0.6)
                        or (
                            intersection_config.is_boundary
                            and outgoing_load >= max(4.0, queue_total * 0.4)
                        )
                    ),
                    "incident_proximity": intersection_id in self._incident_intersections,
                    "corridor_alignment": self._compute_corridor_alignment(
                        ns_queue=ns_queue_local,
                        ew_queue=ew_queue_local,
                        ns_wait=ns_wait_local,
                        ew_wait=ew_wait_local,
                    ),
                }
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
        candidate_intersections = self._build_candidate_intersections(
            candidate_seed_items=candidate_seed_items,
            overload_flag=overload_flag,
            event_flag=event_flag,
        )

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
            candidate_intersections=candidate_intersections,
        ).validate()

    @staticmethod
    def _compute_outgoing_load(env, lane_vehicle_count: dict[str, int], intersection_id: str) -> float:
        intersection_config = env.intersections[intersection_id]
        if not intersection_config.outgoing_lanes:
            return 0.0
        return float(
            sum(float(lane_vehicle_count.get(lane_id, 0)) for lane_id in intersection_config.outgoing_lanes)
        )

    @staticmethod
    def _compute_corridor_alignment(
        ns_queue: float,
        ew_queue: float,
        ns_wait: float,
        ew_wait: float,
    ) -> str:
        ns_pressure = ns_queue + 1.5 * ns_wait
        ew_pressure = ew_queue + 1.5 * ew_wait
        if ns_pressure > ew_pressure * 1.1:
            return "NS"
        if ew_pressure > ns_pressure * 1.1:
            return "EW"
        return "BALANCED"

    @staticmethod
    def _load_road_endpoints(roadnet_path: Path) -> dict[str, tuple[str, str]]:
        roadnet = load_json(roadnet_path)
        return {
            str(road["id"]): (
                str(road["startIntersection"]),
                str(road["endIntersection"]),
            )
            for road in roadnet.get("roads", [])
        }

    def _derive_incident_intersections(self) -> set[str]:
        if not self._road_endpoints:
            return set()
        scenario_metadata = self._scenario_metadata or {}
        details = scenario_metadata.get("details", {})
        incident_roads = list(scenario_metadata.get("blocked_roads", []))
        incident_roads.extend(details.get("accident_roads", []))
        incident_roads.extend(details.get("construction_roads", []))
        if not incident_roads:
            incident_roads.extend(list((scenario_metadata.get("penalized_roads") or {}).keys()))

        intersections: set[str] = set()
        for road_id in incident_roads:
            endpoints = self._road_endpoints.get(str(road_id))
            if endpoints is None:
                continue
            intersections.update(endpoints)
        return intersections

    def _build_candidate_intersections(
        self,
        candidate_seed_items: list[dict[str, Any]],
        overload_flag: bool,
        event_flag: bool,
    ) -> list[CandidateIntersection]:
        if not candidate_seed_items or self.candidate_limit <= 0:
            return []

        def severity_key(item: dict[str, Any]) -> tuple[float, float, float, float, str]:
            candidate = CandidateIntersection(
                intersection_id=str(item["intersection_id"]),
                queue_total=float(item["queue_total"]),
                wait_total=float(item["wait_total"]),
                outgoing_load=float(item["outgoing_load"]),
                current_phase=int(item["current_phase"]),
                is_boundary=bool(item["is_boundary"]),
                spillback_risk=bool(item["spillback_risk"]),
                incident_proximity=bool(item["incident_proximity"]),
                overload_marker=overload_flag,
                event_proximity=event_flag,
                corridor_alignment=str(item["corridor_alignment"]),
                selection_reasons=[],
            )
            return (
                candidate_priority_score(candidate),
                float(item["queue_total"]),
                float(item["wait_total"]),
                float(item["outgoing_load"]),
                str(item["intersection_id"]),
            )

        overall_sorted = sorted(
            candidate_seed_items,
            key=lambda item: (
                -severity_key(item)[0],
                -severity_key(item)[1],
                -severity_key(item)[2],
                -severity_key(item)[3],
                severity_key(item)[4],
            ),
        )
        boundary_sorted = [item for item in overall_sorted if item["is_boundary"]]
        spillback_sorted = [item for item in overall_sorted if item["spillback_risk"]]
        incident_sorted = [item for item in overall_sorted if item["incident_proximity"]]
        outgoing_sorted = sorted(
            candidate_seed_items,
            key=lambda item: (
                -float(item["outgoing_load"]),
                -float(item["queue_total"]),
                -float(item["wait_total"]),
                str(item["intersection_id"]),
            ),
        )

        reason_tags: dict[str, set[str]] = {}
        selected_ids: list[str] = []

        def mark(items: list[dict[str, Any]], tag: str, limit: int) -> None:
            for item in items[:limit]:
                intersection_id = str(item["intersection_id"])
                reason_tags.setdefault(intersection_id, set()).add(tag)
                if intersection_id not in selected_ids:
                    selected_ids.append(intersection_id)

        mark(overall_sorted, "congested", max(1, min(self.top_k, self.candidate_limit)))
        mark(boundary_sorted, "boundary", min(2, self.candidate_limit))
        mark(spillback_sorted, "spillback", min(2, self.candidate_limit))
        mark(incident_sorted, "incident", min(2, self.candidate_limit))
        mark(outgoing_sorted, "outgoing", min(2, self.candidate_limit))
        if overload_flag:
            mark(overall_sorted, "overload", min(2, self.candidate_limit))
        if event_flag:
            event_seed = boundary_sorted if boundary_sorted else outgoing_sorted
            mark(event_seed, "event", min(2, self.candidate_limit))

        for item in overall_sorted:
            if len(selected_ids) >= self.candidate_limit:
                break
            intersection_id = str(item["intersection_id"])
            if intersection_id in selected_ids:
                continue
            selected_ids.append(intersection_id)
            reason_tags.setdefault(intersection_id, {"congested"})

        seed_lookup = {
            str(item["intersection_id"]): item
            for item in candidate_seed_items
        }
        candidates = [
            CandidateIntersection(
                intersection_id=intersection_id,
                queue_total=float(seed_lookup[intersection_id]["queue_total"]),
                wait_total=float(seed_lookup[intersection_id]["wait_total"]),
                outgoing_load=float(seed_lookup[intersection_id]["outgoing_load"]),
                current_phase=int(seed_lookup[intersection_id]["current_phase"]),
                is_boundary=bool(seed_lookup[intersection_id]["is_boundary"]),
                spillback_risk=bool(seed_lookup[intersection_id]["spillback_risk"]),
                incident_proximity=bool(seed_lookup[intersection_id]["incident_proximity"]),
                overload_marker=overload_flag,
                event_proximity=event_flag,
                corridor_alignment=str(seed_lookup[intersection_id]["corridor_alignment"]),
                selection_reasons=sorted(reason_tags.get(intersection_id, {"congested"})),
            ).validate()
            for intersection_id in selected_ids[: self.candidate_limit]
        ]
        return sorted(
            candidates,
            key=lambda item: (
                -candidate_priority_score(item),
                -item.queue_total,
                -item.wait_total,
                -item.outgoing_load,
                item.intersection_id,
            ),
        )
