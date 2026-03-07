from __future__ import annotations

from dataclasses import dataclass

DISTRICT_TYPES: tuple[str, ...] = (
    "residential",
    "commercial",
    "industrial",
    "mixed",
)
DISTRICT_TYPE_TO_INDEX: dict[str, int] = {
    district_type: index for index, district_type in enumerate(DISTRICT_TYPES)
}
DEFAULT_DISTRICT_TYPE = "mixed"


@dataclass(frozen=True)
class PhaseConfig:
    engine_phase_index: int
    available_road_links: tuple[int, ...]
    incoming_lanes_served: tuple[str, ...]
    outgoing_lanes_served: tuple[str, ...]


@dataclass(frozen=True)
class IntersectionConfig:
    intersection_id: str
    district_id: str
    district_type: str
    district_type_index: int
    incoming_lanes: tuple[str, ...]
    outgoing_lanes: tuple[str, ...]
    is_boundary: bool
    green_phases: tuple[PhaseConfig, ...]
    all_phase_indices: tuple[int, ...]
    initial_engine_phase_index: int

    @property
    def num_green_phases(self) -> int:
        return len(self.green_phases)


@dataclass(frozen=True)
class DistrictConfig:
    district_id: str
    district_type: str
    district_type_index: int
    intersection_ids: tuple[str, ...]
    neighbor_districts: tuple[str, ...]
