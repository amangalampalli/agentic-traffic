from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


VALID_MODES = {
    "none",
    "prioritize_ns",
    "prioritize_ew",
    "green_wave",
    "emergency_route",
    "damp_border_inflow",
}


@dataclass
class NeighborMessage:
    sender_intersection: str
    receiver_intersection: str
    congestion_level: float
    spillback_risk: bool
    dominant_direction: str  # "ns", "ew", or "balanced"
    queue_total: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DistrictDirective:
    mode: str = "none"
    target_intersections: list[str] = field(default_factory=list)
    duration: int = 1
    rationale: str = ""
    corridor: str | None = None
    district_weight: float = 0.5

    def validate(self) -> "DistrictDirective":
        if self.mode not in VALID_MODES:
            self.mode = "none"

        if not isinstance(self.target_intersections, list):
            self.target_intersections = []

        if not isinstance(self.duration, int):
            self.duration = 1
        self.duration = max(1, min(self.duration, 10))

        if not isinstance(self.rationale, str):
            self.rationale = ""

        if self.corridor is not None and self.corridor not in {
            "ns",
            "ew",
            "west_to_east",
            "east_to_west",
            "north_to_south",
            "south_to_north",
        }:
            self.corridor = None

        if not isinstance(self.district_weight, (int, float)):
            self.district_weight = 0.5
        self.district_weight = float(max(0.0, min(1.0, self.district_weight)))

        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_district_directive(payload: str | dict[str, Any]) -> DistrictDirective:
    """
    Accept either raw JSON text or a dict and return a validated DistrictDirective.
    Falls back safely to a no-op directive.
    """
    try:
        if isinstance(payload, str):
            payload = payload.strip()
            if not payload:
                return DistrictDirective().validate()

            # Try direct JSON parse
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                # Try to extract JSON object from surrounding text
                start = payload.find("{")
                end = payload.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    return DistrictDirective().validate()
                data = json.loads(payload[start : end + 1])
        elif isinstance(payload, dict):
            data = payload
        else:
            return DistrictDirective().validate()

        directive = DistrictDirective(
            mode=data.get("mode", "none"),
            target_intersections=data.get("target_intersections", []),
            duration=data.get("duration", 1),
            rationale=data.get("rationale", ""),
            corridor=data.get("corridor"),
            district_weight=data.get("district_weight", 0.5),
        )
        return directive.validate()
    except Exception:
        return DistrictDirective().validate()


def safe_directive_dict(payload: str | dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return DistrictDirective().validate().to_dict()
    return parse_district_directive(payload).to_dict()
