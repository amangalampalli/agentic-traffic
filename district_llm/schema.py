from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


DISTRICT_STRATEGIES: tuple[str, ...] = (
    "hold",
    "favor_NS",
    "favor_EW",
    "drain_inbound",
    "drain_outbound",
    "clear_spillback",
    "incident_response",
    "arterial_priority",
)
PHASE_BIASES: tuple[str, ...] = ("NONE", "NS", "EW")
PRIORITY_CORRIDORS: tuple[str, ...] = (
    "NS",
    "EW",
    "inbound",
    "outbound",
    "arterial",
)
DOMINANT_FLOWS: tuple[str, ...] = ("NS", "EW", "BALANCED")


def _round_float(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _stable_string_list(values: list[str] | tuple[str, ...] | None, limit: int | None = None) -> list[str]:
    normalized = sorted({str(item) for item in (values or []) if str(item).strip()})
    if limit is not None:
        normalized = normalized[:limit]
    return normalized


@dataclass
class CongestedIntersection:
    intersection_id: str
    queue_total: float
    wait_total: float
    outgoing_load: float
    current_phase: int
    is_boundary: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "intersection_id": self.intersection_id,
            "queue_total": _round_float(self.queue_total),
            "wait_total": _round_float(self.wait_total),
            "outgoing_load": _round_float(self.outgoing_load),
            "current_phase": int(self.current_phase),
            "is_boundary": bool(self.is_boundary),
        }

    def to_prompt_line(self) -> str:
        return (
            f"- {self.intersection_id} "
            f"q={self.queue_total:.2f} "
            f"w={self.wait_total:.2f} "
            f"out={self.outgoing_load:.2f} "
            f"phase={self.current_phase} "
            f"boundary={int(self.is_boundary)}"
        )


@dataclass
class DistrictAction:
    strategy: str = "hold"
    priority_corridor: str | None = None
    target_intersections: list[str] = field(default_factory=list)
    phase_bias: str = "NONE"
    duration_steps: int = 1

    def validate(self) -> "DistrictAction":
        if self.strategy not in DISTRICT_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. Expected one of {DISTRICT_STRATEGIES}."
            )
        if self.priority_corridor is not None and self.priority_corridor not in PRIORITY_CORRIDORS:
            raise ValueError(
                f"Invalid priority_corridor '{self.priority_corridor}'. "
                f"Expected one of {PRIORITY_CORRIDORS} or None."
            )
        if self.phase_bias not in PHASE_BIASES:
            raise ValueError(
                f"Invalid phase_bias '{self.phase_bias}'. Expected one of {PHASE_BIASES}."
            )
        if not isinstance(self.duration_steps, int):
            raise ValueError("duration_steps must be an integer.")
        if not 1 <= self.duration_steps <= 20:
            raise ValueError("duration_steps must be between 1 and 20.")
        self.target_intersections = _stable_string_list(self.target_intersections, limit=8)
        return self

    @classmethod
    def default_hold(cls, duration_steps: int = 1) -> "DistrictAction":
        return cls(
            strategy="hold",
            priority_corridor=None,
            target_intersections=[],
            phase_bias="NONE",
            duration_steps=max(1, min(int(duration_steps), 20)),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DistrictAction":
        return cls(
            strategy=str(payload.get("strategy", "hold")),
            priority_corridor=payload.get("priority_corridor"),
            target_intersections=list(payload.get("target_intersections", [])),
            phase_bias=str(payload.get("phase_bias", "NONE")),
            duration_steps=int(payload.get("duration_steps", 1)),
        ).validate()

    @classmethod
    def from_json(cls, payload: str) -> "DistrictAction":
        return cls.from_dict(json.loads(payload))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "strategy": self.strategy,
            "priority_corridor": self.priority_corridor,
            "target_intersections": list(self.target_intersections),
            "phase_bias": self.phase_bias,
            "duration_steps": int(self.duration_steps),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_pretty_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def to_rl_context(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["district_strategy"] = payload.pop("strategy")
        payload["district_duration_steps"] = payload.pop("duration_steps")
        return payload


@dataclass
class DistrictStateSummary:
    city_id: str
    district_id: str
    district_type: str
    scenario_name: str
    scenario_type: str
    decision_step: int
    sim_time: int
    intersection_count: int
    avg_queue: float
    max_queue: float
    total_queue: float
    avg_wait: float
    max_wait: float
    total_wait: float
    avg_outgoing_load: float
    max_outgoing_load: float
    total_outgoing_load: float
    recent_throughput: float
    queue_change: float
    wait_change: float
    throughput_change: float
    ns_queue: float
    ew_queue: float
    ns_wait: float
    ew_wait: float
    dominant_flow: str
    boundary_queue_total: float
    boundary_wait_total: float
    spillback_risk: bool
    incident_flag: bool
    construction_flag: bool
    overload_flag: bool
    event_flag: bool
    top_congested_intersections: list[CongestedIntersection] = field(default_factory=list)

    def validate(self) -> "DistrictStateSummary":
        if self.dominant_flow not in DOMINANT_FLOWS:
            raise ValueError(
                f"Invalid dominant_flow '{self.dominant_flow}'. Expected one of {DOMINANT_FLOWS}."
            )
        self.top_congested_intersections = list(self.top_congested_intersections[:5])
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "city_id": self.city_id,
            "district_id": self.district_id,
            "district_type": self.district_type,
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type,
            "decision_step": int(self.decision_step),
            "sim_time": int(self.sim_time),
            "intersection_count": int(self.intersection_count),
            "avg_queue": _round_float(self.avg_queue),
            "max_queue": _round_float(self.max_queue),
            "total_queue": _round_float(self.total_queue),
            "avg_wait": _round_float(self.avg_wait),
            "max_wait": _round_float(self.max_wait),
            "total_wait": _round_float(self.total_wait),
            "avg_outgoing_load": _round_float(self.avg_outgoing_load),
            "max_outgoing_load": _round_float(self.max_outgoing_load),
            "total_outgoing_load": _round_float(self.total_outgoing_load),
            "recent_throughput": _round_float(self.recent_throughput),
            "queue_change": _round_float(self.queue_change),
            "wait_change": _round_float(self.wait_change),
            "throughput_change": _round_float(self.throughput_change),
            "ns_queue": _round_float(self.ns_queue),
            "ew_queue": _round_float(self.ew_queue),
            "ns_wait": _round_float(self.ns_wait),
            "ew_wait": _round_float(self.ew_wait),
            "dominant_flow": self.dominant_flow,
            "boundary_queue_total": _round_float(self.boundary_queue_total),
            "boundary_wait_total": _round_float(self.boundary_wait_total),
            "spillback_risk": bool(self.spillback_risk),
            "incident_flag": bool(self.incident_flag),
            "construction_flag": bool(self.construction_flag),
            "overload_flag": bool(self.overload_flag),
            "event_flag": bool(self.event_flag),
            "top_congested_intersections": [
                item.to_dict() for item in self.top_congested_intersections
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_prompt_text(self) -> str:
        self.validate()
        top_lines = [item.to_prompt_line() for item in self.top_congested_intersections]
        if not top_lines:
            top_lines = ["- none"]
        return "\n".join(
            [
                f"city_id: {self.city_id}",
                f"district_id: {self.district_id}",
                f"district_type: {self.district_type}",
                f"scenario: {self.scenario_name}",
                f"scenario_type: {self.scenario_type}",
                f"decision_step: {self.decision_step}",
                f"sim_time: {self.sim_time}",
                f"intersection_count: {self.intersection_count}",
                f"avg_queue: {self.avg_queue:.2f}",
                f"max_queue: {self.max_queue:.2f}",
                f"total_queue: {self.total_queue:.2f}",
                f"avg_wait: {self.avg_wait:.2f}",
                f"max_wait: {self.max_wait:.2f}",
                f"total_wait: {self.total_wait:.2f}",
                f"avg_outgoing_load: {self.avg_outgoing_load:.2f}",
                f"max_outgoing_load: {self.max_outgoing_load:.2f}",
                f"total_outgoing_load: {self.total_outgoing_load:.2f}",
                f"recent_throughput: {self.recent_throughput:.2f}",
                f"queue_change: {self.queue_change:.2f}",
                f"wait_change: {self.wait_change:.2f}",
                f"throughput_change: {self.throughput_change:.2f}",
                f"ns_queue: {self.ns_queue:.2f}",
                f"ew_queue: {self.ew_queue:.2f}",
                f"ns_wait: {self.ns_wait:.2f}",
                f"ew_wait: {self.ew_wait:.2f}",
                f"dominant_flow: {self.dominant_flow}",
                f"boundary_queue_total: {self.boundary_queue_total:.2f}",
                f"boundary_wait_total: {self.boundary_wait_total:.2f}",
                f"spillback_risk: {int(self.spillback_risk)}",
                f"incident_flag: {int(self.incident_flag)}",
                f"construction_flag: {int(self.construction_flag)}",
                f"overload_flag: {int(self.overload_flag)}",
                f"event_flag: {int(self.event_flag)}",
                "top_congested_intersections:",
                *top_lines,
            ]
        )
