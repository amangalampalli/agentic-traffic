from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from district_llm.schema import DistrictAction, DistrictStateSummary


@dataclass
class LocalIntersectionAction:
    intersection_id: str
    district_id: str
    action: int
    current_phase: int
    next_phase: int
    queue_total: float
    wait_total: float
    outgoing_load: float
    is_boundary: bool

    @property
    def switched(self) -> bool:
        return int(self.action) == 1 and self.next_phase != self.current_phase


@dataclass
class DistrictWindowData:
    district_id: str
    start_summary: DistrictStateSummary
    end_summary: DistrictStateSummary
    controller_actions: list[LocalIntersectionAction] = field(default_factory=list)
    step_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "district_id": self.district_id,
            "step_count": int(self.step_count),
            "queue_delta": round(self.end_summary.total_queue - self.start_summary.total_queue, 3),
            "wait_delta": round(self.end_summary.total_wait - self.start_summary.total_wait, 3),
            "throughput_delta": round(
                self.end_summary.recent_throughput - self.start_summary.recent_throughput,
                3,
            ),
        }


def derive_district_action(
    window_data: DistrictWindowData,
    controller_actions: list[LocalIntersectionAction] | None = None,
    district_state: DistrictStateSummary | None = None,
) -> DistrictAction:
    """
    Deterministic first-pass label extraction from local-controller behavior.

    Heuristic order:
    1. Incident-heavy windows map to `incident_response`.
    2. Strong spillback / boundary pressure maps to `clear_spillback`.
    3. Rising boundary demand maps to `drain_inbound`.
    4. Persistently high outgoing pressure maps to `drain_outbound`.
    5. Boundary-heavy rush windows map to `arterial_priority`.
    6. Clear NS/EW directional dominance maps to `favor_NS` / `favor_EW`.
    7. Otherwise emit `hold`.
    """
    actions = controller_actions if controller_actions is not None else window_data.controller_actions
    state = district_state if district_state is not None else window_data.start_summary
    end_state = window_data.end_summary

    duration_steps = max(1, min(int(window_data.step_count or 1), 20))
    phase_counts = {"NS": 0, "EW": 0}
    focus_scores: dict[str, float] = {}
    boundary_focus = 0
    switch_count = 0

    for item in actions:
        phase_key = "NS" if int(item.next_phase) == 0 else "EW"
        phase_counts[phase_key] += 1
        switch_count += int(item.switched)
        if item.is_boundary:
            boundary_focus += 1
        focus_scores[item.intersection_id] = focus_scores.get(item.intersection_id, 0.0) + (
            item.queue_total + 1.5 * item.wait_total + 2.0 * float(item.switched)
        )

    total_action_records = max(1, len(actions))
    ns_phase_ratio = phase_counts["NS"] / float(total_action_records)
    ew_phase_ratio = phase_counts["EW"] / float(total_action_records)
    boundary_focus_ratio = boundary_focus / float(total_action_records)
    queue_delta = end_state.total_queue - state.total_queue
    wait_delta = end_state.total_wait - state.total_wait
    boundary_share = state.boundary_queue_total / max(1.0, state.total_queue)
    outgoing_pressure = end_state.total_outgoing_load / max(1.0, end_state.total_queue)

    if ns_phase_ratio > ew_phase_ratio + 0.1:
        phase_bias = "NS"
    elif ew_phase_ratio > ns_phase_ratio + 0.1:
        phase_bias = "EW"
    else:
        phase_bias = "NONE"

    if phase_bias == "NONE" and state.dominant_flow in {"NS", "EW"}:
        phase_bias = state.dominant_flow

    target_intersections = [
        intersection_id
        for intersection_id, _ in sorted(
            focus_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
    ]
    if not target_intersections:
        target_intersections = [
            item.intersection_id for item in state.top_congested_intersections[:3]
        ]

    if state.incident_flag or end_state.incident_flag:
        return DistrictAction(
            strategy="incident_response",
            priority_corridor=phase_bias if phase_bias in {"NS", "EW"} else "arterial",
            target_intersections=target_intersections,
            phase_bias=phase_bias,
            duration_steps=duration_steps,
        ).validate()

    if state.spillback_risk or end_state.spillback_risk or (boundary_share >= 0.55 and outgoing_pressure >= 0.45):
        return DistrictAction(
            strategy="clear_spillback",
            priority_corridor="inbound" if boundary_share >= 0.55 else phase_bias if phase_bias in {"NS", "EW"} else None,
            target_intersections=target_intersections,
            phase_bias=phase_bias,
            duration_steps=duration_steps,
        ).validate()

    if boundary_share >= 0.55 and (queue_delta >= 0.0 or wait_delta >= 0.0):
        return DistrictAction(
            strategy="drain_inbound",
            priority_corridor="inbound",
            target_intersections=target_intersections,
            phase_bias=phase_bias,
            duration_steps=duration_steps,
        ).validate()

    if outgoing_pressure >= 0.65 and end_state.total_queue >= state.total_queue * 0.9:
        return DistrictAction(
            strategy="drain_outbound",
            priority_corridor="outbound",
            target_intersections=target_intersections,
            phase_bias=phase_bias,
            duration_steps=duration_steps,
        ).validate()

    if (
        state.event_flag
        or state.overload_flag
        or end_state.overload_flag
        or (boundary_focus_ratio >= 0.6 and switch_count >= max(2, duration_steps))
    ):
        return DistrictAction(
            strategy="arterial_priority",
            priority_corridor=phase_bias if phase_bias in {"NS", "EW"} else "arterial",
            target_intersections=target_intersections,
            phase_bias=phase_bias,
            duration_steps=duration_steps,
        ).validate()

    ns_pressure = state.ns_queue + 1.5 * state.ns_wait
    ew_pressure = state.ew_queue + 1.5 * state.ew_wait
    imbalance_threshold = max(5.0, 0.15 * max(1.0, ns_pressure + ew_pressure))

    if ns_pressure - ew_pressure >= imbalance_threshold:
        return DistrictAction(
            strategy="favor_NS",
            priority_corridor="NS",
            target_intersections=target_intersections,
            phase_bias="NS",
            duration_steps=duration_steps,
        ).validate()

    if ew_pressure - ns_pressure >= imbalance_threshold:
        return DistrictAction(
            strategy="favor_EW",
            priority_corridor="EW",
            target_intersections=target_intersections,
            phase_bias="EW",
            duration_steps=duration_steps,
        ).validate()

    return DistrictAction.default_hold(duration_steps=duration_steps)
