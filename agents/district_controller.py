from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from agents.message_protocol import DistrictDirective, parse_district_directive


class BaseDistrictCoordinator(ABC):
    @abstractmethod
    def decide(self, district_summary: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class RuleBasedDistrictCoordinator(BaseDistrictCoordinator):
    """
    Fast, deterministic, and robust.
    Good first coordinator and good fallback if the LLM output fails.
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.15,
        border_pressure_threshold: float = 0.65,
        default_duration: int = 2,
    ):
        self.imbalance_threshold = imbalance_threshold
        self.border_pressure_threshold = border_pressure_threshold
        self.default_duration = default_duration

    def decide(self, district_summary: dict[str, Any]) -> dict[str, Any]:
        district_id = district_summary.get("district_id", "unknown")
        intersection_ids = district_summary.get("intersection_ids", [])

        emergency = district_summary.get("emergency_vehicle", {})
        if emergency.get("present", False):
            return (
                DistrictDirective(
                    mode="emergency_route",
                    target_intersections=emergency.get("route", intersection_ids),
                    duration=2,
                    rationale=f"Emergency vehicle detected in district {district_id}.",
                    corridor=emergency.get("corridor"),
                    district_weight=1.0,
                )
                .validate()
                .to_dict()
            )

        corridor_loads = district_summary.get("corridor_loads", {})
        ns = float(corridor_loads.get("ns", corridor_loads.get("north_south", 0.0)))
        ew = float(corridor_loads.get("ew", corridor_loads.get("east_west", 0.0)))

        border_pressure = district_summary.get("border_pressure", {})
        border_max = 0.0
        if isinstance(border_pressure, dict) and border_pressure:
            border_max = max(float(v) for v in border_pressure.values())

        if ew - ns > self.imbalance_threshold:
            return (
                DistrictDirective(
                    mode="prioritize_ew",
                    target_intersections=intersection_ids,
                    duration=self.default_duration,
                    rationale="East-west corridor is currently more congested than north-south.",
                    corridor="ew",
                    district_weight=(
                        0.7 if border_max < self.border_pressure_threshold else 0.9
                    ),
                )
                .validate()
                .to_dict()
            )

        if ns - ew > self.imbalance_threshold:
            return (
                DistrictDirective(
                    mode="prioritize_ns",
                    target_intersections=intersection_ids,
                    duration=self.default_duration,
                    rationale="North-south corridor is currently more congested than east-west.",
                    corridor="ns",
                    district_weight=(
                        0.7 if border_max < self.border_pressure_threshold else 0.9
                    ),
                )
                .validate()
                .to_dict()
            )

        if border_max >= self.border_pressure_threshold:
            return (
                DistrictDirective(
                    mode="damp_border_inflow",
                    target_intersections=intersection_ids,
                    duration=2,
                    rationale="Border pressure is high; reduce spill-in and smooth cross-district flow.",
                    district_weight=0.8,
                )
                .validate()
                .to_dict()
            )

        return (
            DistrictDirective(
                mode="none",
                target_intersections=[],
                duration=1,
                rationale="District is reasonably balanced.",
                district_weight=0.5,
            )
            .validate()
            .to_dict()
        )


class LLMDistrictCoordinator(BaseDistrictCoordinator):
    """
    LLM-backed coordinator.

    `generator_fn` should accept a prompt string and return either:
      - a JSON string, or
      - a dict

    Example:
        coordinator = LLMDistrictCoordinator(generator_fn=my_model_call)
    """

    def __init__(
        self,
        generator_fn: Callable[[str], str | dict[str, Any]],
        fallback: BaseDistrictCoordinator | None = None,
        max_prompt_chars: int = 4000,
    ):
        self.generator_fn = generator_fn
        self.fallback = fallback or RuleBasedDistrictCoordinator()
        self.max_prompt_chars = max_prompt_chars

    def decide(self, district_summary: dict[str, Any]) -> dict[str, Any]:
        prompt = self.build_prompt(district_summary)
        try:
            raw = self.generator_fn(prompt)
            directive = parse_district_directive(raw).to_dict()

            # If the LLM returns a no-op too often or malformed content,
            # the parser still makes it safe. We keep that behavior.
            return directive
        except Exception:
            return self.fallback.decide(district_summary)

    def build_prompt(self, district_summary: dict[str, Any]) -> str:
        summary_text = repr(district_summary)
        if len(summary_text) > self.max_prompt_chars:
            summary_text = summary_text[: self.max_prompt_chars] + " ...[truncated]"

        return f"""You are a district-level traffic coordinator.

Your job is to choose a single strategic directive for the next few cycles.

Allowed modes:
- none
- prioritize_ns
- prioritize_ew
- green_wave
- emergency_route
- damp_border_inflow

Return ONLY valid JSON with these fields:
{{
  "mode": string,
  "target_intersections": list[string],
  "duration": int,
  "rationale": string,
  "corridor": string or null,
  "district_weight": float
}}

Guidelines:
- Use emergency_route if an emergency vehicle is present.
- Use prioritize_ns or prioritize_ew when one corridor is clearly more congested.
- Use damp_border_inflow when cross-district border pressure is high.
- Keep duration between 1 and 5.
- district_weight should be between 0.0 and 1.0.

District summary:
{summary_text}
"""
