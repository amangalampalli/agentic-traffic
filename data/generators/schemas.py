"""Typed schemas and dataclasses for synthetic CityFlow generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TopologyType = Literal[
    "rectangular_grid",
    "irregular_grid",
    "arterial_local",
    "ring_road",
    "mixed",
]

DistrictType = Literal["residential", "commercial", "industrial", "mixed"]
DemandIntensity = Literal[
    "normal",
    "moderate_rush",
    "heavy_rush",
    "overload",
    "accident_overload",
]
ScenarioType = Literal[
    "normal",
    "morning_rush",
    "evening_rush",
    "accident",
    "construction",
    "event_spike",
    "district_overload",
]


@dataclass(slots=True, frozen=True)
class TripMix:
    """Trip category distribution."""

    intra_district: float = 0.5
    adjacent_district: float = 0.3
    long_distance: float = 0.2


@dataclass(slots=True)
class DatasetGenerationConfig:
    """Top-level CLI / generation configuration."""

    num_cities: int
    output_dir: Path
    seed: int = 42
    min_districts: int = 6
    max_districts: int = 20
    min_intersections_per_district: int = 4
    max_intersections_per_district: int = 10
    topologies: list[TopologyType] = field(
        default_factory=lambda: ["irregular_grid"]
    )
    scenarios: list[ScenarioType] = field(
        default_factory=lambda: [
            "normal",
            "morning_rush",
            "evening_rush",
            "accident",
            "construction",
            "event_spike",
            "district_overload",
        ]
    )
    intensity_levels: list[DemandIntensity] = field(
        default_factory=lambda: [
            "normal",
            "moderate_rush",
            "heavy_rush",
            "overload",
            "accident_overload",
        ]
    )
    intensity_distribution: dict[DemandIntensity, float] = field(
        default_factory=lambda: {
            "normal": 0.20,
            "moderate_rush": 0.42,
            "heavy_rush": 0.24,
            "overload": 0.10,
            "accident_overload": 0.04,
        }
    )
    global_demand_multiplier: float = 1.25
    scenario_demand_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "normal": 1.15,
            "morning_rush": 1.35,
            "evening_rush": 1.35,
            "accident": 1.75,
            "construction": 1.55,
            "event_spike": 1.65,
            "district_overload": 1.70,
        }
    )
    ring_diagonal_keep_prob: float = 0.07
    ring_max_diagonal_fraction: float = 0.03
    simulation_steps: int = 3600
    interval: float = 1.0
    save_replay: bool = False
    fail_fast: bool = False


@dataclass(slots=True, frozen=True)
class RoadRecord:
    """Directed road edge record."""

    id: str
    start_intersection: str
    end_intersection: str
    length: float
    speed_limit: float
    num_lanes: int
    points: list[dict[str, float]]
    is_arterial: bool


@dataclass(slots=True)
class CityGraph:
    """Intermediate graph representation for generation."""

    city_id: str
    topology: TopologyType
    seed: int
    intersections: dict[str, tuple[float, float]]
    adjacency: dict[str, set[str]]
    directed_roads: dict[str, RoadRecord]
    roadnet: dict[str, Any]
    arterial_roads: set[str]
    gateway_intersections: set[str] = field(default_factory=set)
    gateway_roads: set[str] = field(default_factory=set)
    inter_district_roads: set[str] = field(default_factory=set)


@dataclass(slots=True)
class DistrictRecord:
    """District-level metadata."""

    id: str
    district_type: DistrictType
    intersections: list[str]
    neighbors: list[str]
    boundary_intersections: list[str]
    entry_roads: list[str]
    exit_roads: list[str]


@dataclass(slots=True)
class DistrictData:
    """District overlay output."""

    intersection_to_district: dict[str, str]
    districts: dict[str, DistrictRecord]
    district_neighbors: dict[str, list[str]]
    boundary_intersections: list[str]
    inter_district_roads: list[str]


@dataclass(slots=True)
class ScenarioPlan:
    """Scenario-specific route demand and impairment configuration."""

    name: ScenarioType
    intensity: DemandIntensity
    seed: int
    trip_multiplier: float
    trip_mix: TripMix
    departure_windows: list[tuple[float, float, float]]
    blocked_roads: set[str] = field(default_factory=set)
    penalized_roads: dict[str, float] = field(default_factory=dict)
    event_district: str | None = None
    overload_district: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
