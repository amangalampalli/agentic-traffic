from __future__ import annotations

from dataclasses import dataclass

from district_llm.repair import fallback_target_intersections
from district_llm.schema import DistrictAction, DistrictStateSummary


@dataclass(frozen=True)
class HeuristicGuidanceConfig:
    max_target_intersections: int = 3
    incident_duration_steps: int = 12
    spillback_duration_steps: int = 10
    default_duration_steps: int = 8


def generate_heuristic_guidance(
    summary: DistrictStateSummary,
    config: HeuristicGuidanceConfig | None = None,
) -> DistrictAction:
    config = config or HeuristicGuidanceConfig()

    if summary.incident_flag or summary.construction_flag:
        strategy = "incident_response"
        priority_corridor = summary.dominant_flow if summary.dominant_flow in {"NS", "EW"} else "arterial"
        phase_bias = summary.dominant_flow if summary.dominant_flow in {"NS", "EW"} else "NONE"
        duration_steps = config.incident_duration_steps
    elif summary.spillback_risk:
        strategy = "clear_spillback"
        boundary_share = summary.boundary_queue_total / max(1.0, summary.total_queue)
        if boundary_share >= 0.45:
            priority_corridor = "inbound"
        elif summary.dominant_flow in {"NS", "EW"}:
            priority_corridor = summary.dominant_flow
        else:
            priority_corridor = None
        phase_bias = summary.dominant_flow if summary.dominant_flow in {"NS", "EW"} else "NONE"
        duration_steps = config.spillback_duration_steps
    elif summary.event_flag or summary.overload_flag:
        strategy = "arterial_priority"
        priority_corridor = summary.dominant_flow if summary.dominant_flow in {"NS", "EW"} else "arterial"
        phase_bias = summary.dominant_flow if summary.dominant_flow in {"NS", "EW"} else "NONE"
        duration_steps = config.spillback_duration_steps
    elif summary.dominant_flow == "NS":
        strategy = "favor_NS"
        priority_corridor = "NS"
        phase_bias = "NS"
        duration_steps = config.default_duration_steps
    elif summary.dominant_flow == "EW":
        strategy = "favor_EW"
        priority_corridor = "EW"
        phase_bias = "EW"
        duration_steps = config.default_duration_steps
    else:
        strategy = "hold"
        priority_corridor = None
        phase_bias = "NONE"
        duration_steps = config.default_duration_steps

    targets = fallback_target_intersections(
        summary=summary,
        max_target_intersections=config.max_target_intersections,
        strategy=strategy,
        priority_corridor=priority_corridor,
        phase_bias=phase_bias,
    )
    return DistrictAction(
        strategy=strategy,
        priority_corridor=priority_corridor,
        target_intersections=targets,
        phase_bias=phase_bias,
        duration_steps=duration_steps,
    ).validate()
