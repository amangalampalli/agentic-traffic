from __future__ import annotations

import json
import math
from pathlib import Path

from env.intersection_config import (
    DEFAULT_DISTRICT_TYPE,
    DISTRICT_TYPE_TO_INDEX,
    DistrictConfig,
    IntersectionConfig,
    PhaseConfig,
)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize_scalar(value: float, scale: float) -> float:
    if scale <= 0:
        return float(value)
    return float(value) / float(scale)


def lane_ids_for_road(road: dict) -> tuple[str, ...]:
    return tuple(f"{road['id']}_{lane_index}" for lane_index, _ in enumerate(road["lanes"]))


def build_topology(
    roadnet_path: str | Path,
    district_map_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[dict[str, IntersectionConfig], dict[str, DistrictConfig]]:
    roadnet = load_json(roadnet_path)
    district_map = load_json(district_map_path) if district_map_path else {}
    metadata = load_json(metadata_path) if metadata_path else {}

    intersection_to_district = district_map.get("intersection_to_district", {})
    district_neighbors = district_map.get("district_neighbors", {})
    district_types = metadata.get("district_types", {})

    roads = {road["id"]: road for road in roadnet["roads"]}
    road_lookup_by_end: dict[str, list[dict]] = {}
    road_lookup_by_start: dict[str, list[dict]] = {}
    for road in roadnet["roads"]:
        road_lookup_by_end.setdefault(road["endIntersection"], []).append(road)
        road_lookup_by_start.setdefault(road["startIntersection"], []).append(road)

    intersections: dict[str, IntersectionConfig] = {}
    district_to_intersections: dict[str, list[str]] = {}

    for intersection in roadnet["intersections"]:
        if intersection.get("virtual", False):
            continue

        intersection_id = intersection["id"]
        district_id = intersection_to_district.get(intersection_id, "unknown")
        incoming_roads = _sort_roads_around_intersection(
            intersection=intersection,
            roads=road_lookup_by_end.get(intersection_id, []),
            incoming=True,
        )
        outgoing_roads = _sort_roads_around_intersection(
            intersection=intersection,
            roads=road_lookup_by_start.get(intersection_id, []),
            incoming=False,
        )

        incoming_lanes = tuple(
            lane_id
            for road in incoming_roads
            for lane_id in lane_ids_for_road(road)
        )
        outgoing_lanes = tuple(
            lane_id
            for road in outgoing_roads
            for lane_id in lane_ids_for_road(road)
        )

        green_phases: list[PhaseConfig] = []
        lightphases = intersection.get("trafficLight", {}).get("lightphases", [])
        road_links = intersection.get("roadLinks", [])
        for engine_phase_index, phase in enumerate(lightphases):
            available_road_links = tuple(phase.get("availableRoadLinks", []))
            if not available_road_links:
                continue

            served_incoming: set[str] = set()
            served_outgoing: set[str] = set()
            for road_link_index in available_road_links:
                road_link = road_links[road_link_index]
                start_road = road_link["startRoad"]
                end_road = road_link["endRoad"]
                for lane_link in road_link.get("laneLinks", []):
                    served_incoming.add(
                        f"{start_road}_{int(lane_link['startLaneIndex'])}"
                    )
                    served_outgoing.add(
                        f"{end_road}_{int(lane_link['endLaneIndex'])}"
                    )

            green_phases.append(
                PhaseConfig(
                    engine_phase_index=engine_phase_index,
                    available_road_links=available_road_links,
                    incoming_lanes_served=tuple(sorted(served_incoming)),
                    outgoing_lanes_served=tuple(sorted(served_outgoing)),
                )
            )

        if len(green_phases) < 2:
            continue

        district_type = _normalize_district_type(
            district_types.get(district_id, DEFAULT_DISTRICT_TYPE)
        )
        initial_phase_index = (
            green_phases[0].engine_phase_index
            if green_phases
            else 0
        )
        intersections[intersection_id] = IntersectionConfig(
            intersection_id=intersection_id,
            district_id=district_id,
            district_type=district_type,
            district_type_index=DISTRICT_TYPE_TO_INDEX[district_type],
            incoming_lanes=incoming_lanes,
            outgoing_lanes=outgoing_lanes,
            is_boundary=_is_boundary_intersection(
                intersection_id=intersection_id,
                district_id=district_id,
                incoming_roads=incoming_roads,
                outgoing_roads=outgoing_roads,
                intersection_to_district=intersection_to_district,
            ),
            green_phases=tuple(green_phases),
            all_phase_indices=tuple(range(len(lightphases))),
            initial_engine_phase_index=initial_phase_index,
        )
        district_to_intersections.setdefault(district_id, []).append(intersection_id)

    districts: dict[str, DistrictConfig] = {}
    for district_id, intersection_ids in district_to_intersections.items():
        district_type = _normalize_district_type(
            district_types.get(district_id, DEFAULT_DISTRICT_TYPE)
        )
        districts[district_id] = DistrictConfig(
            district_id=district_id,
            district_type=district_type,
            district_type_index=DISTRICT_TYPE_TO_INDEX[district_type],
            intersection_ids=tuple(sorted(intersection_ids)),
            neighbor_districts=tuple(sorted(district_neighbors.get(district_id, []))),
        )

    return intersections, districts


def _sort_roads_around_intersection(
    intersection: dict,
    roads: list[dict],
    incoming: bool,
) -> list[dict]:
    center_x = float(intersection["point"]["x"])
    center_y = float(intersection["point"]["y"])

    def angle_for_road(road: dict) -> tuple[float, str]:
        points = road.get("points", [])
        if not points:
            return (0.0, road["id"])

        reference_point = points[0] if incoming else points[-1]
        dx = float(reference_point["x"]) - center_x
        dy = float(reference_point["y"]) - center_y
        angle = math.atan2(dy, dx)
        return (angle, road["id"])

    return sorted(roads, key=angle_for_road)


def _normalize_district_type(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in DISTRICT_TYPE_TO_INDEX:
        return DEFAULT_DISTRICT_TYPE
    return normalized


def _is_boundary_intersection(
    intersection_id: str,
    district_id: str,
    incoming_roads: list[dict],
    outgoing_roads: list[dict],
    intersection_to_district: dict[str, str],
) -> bool:
    connected_intersections = {
        road["startIntersection"] for road in incoming_roads
    } | {
        road["endIntersection"] for road in outgoing_roads
    }
    connected_intersections.discard(intersection_id)
    for neighbor_intersection_id in connected_intersections:
        neighbor_district_id = intersection_to_district.get(neighbor_intersection_id)
        if neighbor_district_id is not None and neighbor_district_id != district_id:
            return True
    return False
