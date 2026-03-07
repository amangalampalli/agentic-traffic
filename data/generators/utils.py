"""Utilities for deterministic generation, IO, graph operations, and validation."""

from __future__ import annotations

import heapq
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .schemas import CityGraph, DistrictData


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def choose_weighted(rng: random.Random, values: list[str], weights: list[float]) -> str:
    total = sum(weights)
    if total <= 0:
        return values[rng.randrange(len(values))]
    cutoff = rng.random() * total
    cursor = 0.0
    for value, weight in zip(values, weights):
        cursor += weight
        if cursor >= cutoff:
            return value
    return values[-1]


def connected_components(nodes: Iterable[str], adjacency: dict[str, set[str]]) -> list[set[str]]:
    pending = set(nodes)
    components: list[set[str]] = []
    while pending:
        root = pending.pop()
        comp = {root}
        queue = deque([root])
        while queue:
            cur = queue.popleft()
            for nxt in adjacency[cur]:
                if nxt in pending:
                    pending.remove(nxt)
                    comp.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components


def dijkstra_shortest_path(
    start: str,
    end: str,
    graph: dict[str, list[tuple[str, float, str]]],
) -> list[str] | None:
    """Return road-id path from start intersection to end intersection."""
    if start == end:
        return []
    queue: list[tuple[float, str]] = [(0.0, start)]
    dist: dict[str, float] = {start: 0.0}
    prev: dict[str, tuple[str, str]] = {}

    while queue:
        cost, node = heapq.heappop(queue)
        if node == end:
            break
        if cost > dist[node]:
            continue
        for nxt, edge_cost, road_id in graph.get(node, []):
            candidate = cost + edge_cost
            if candidate < dist.get(nxt, float("inf")):
                dist[nxt] = candidate
                prev[nxt] = (node, road_id)
                heapq.heappush(queue, (candidate, nxt))

    if end not in prev:
        return None

    route: list[str] = []
    cursor = end
    while cursor != start:
        parent, road_id = prev[cursor]
        route.append(road_id)
        cursor = parent
    route.reverse()
    return route


def validate_unique_ids(city_graph: CityGraph) -> None:
    intersection_ids = set(city_graph.intersections.keys())
    if len(intersection_ids) != len(city_graph.intersections):
        raise ValueError("Duplicate intersection IDs found.")
    road_ids = set(city_graph.directed_roads.keys())
    if len(road_ids) != len(city_graph.directed_roads):
        raise ValueError("Duplicate road IDs found.")


def validate_district_contiguity(city_graph: CityGraph, district_data: DistrictData) -> None:
    by_district: dict[str, set[str]] = defaultdict(set)
    for intersection_id, district_id in district_data.intersection_to_district.items():
        by_district[district_id].add(intersection_id)

    for district_id, members in by_district.items():
        if not members:
            raise ValueError(f"District {district_id} has no intersections.")
        components = connected_components(members, city_graph.adjacency)
        if len(components) != 1:
            raise ValueError(f"District {district_id} is not contiguous.")


def validate_inter_district_connectivity(district_data: DistrictData) -> None:
    connected = sum(1 for roads in district_data.district_neighbors.values() if roads)
    if connected == 0:
        raise ValueError("No inter-district connections found.")


def validate_district_exit_capacity(
    district_data: DistrictData,
    min_exit_roads: int = 2,
    min_entry_roads: int = 2,
    min_neighbor_districts: int = 1,
) -> None:
    underconnected_exit: list[str] = []
    underconnected_entry: list[str] = []
    underconnected_neighbors: list[str] = []
    for district_id, district in district_data.districts.items():
        if len(district.exit_roads) < min_exit_roads:
            underconnected_exit.append(
                f"{district_id}:{len(district.exit_roads)}"
            )
        if len(district.entry_roads) < min_entry_roads:
            underconnected_entry.append(
                f"{district_id}:{len(district.entry_roads)}"
            )
        if len(district.neighbors) < min_neighbor_districts:
            underconnected_neighbors.append(
                f"{district_id}:{len(district.neighbors)}"
            )
    if underconnected_exit or underconnected_entry or underconnected_neighbors:
        parts: list[str] = []
        if underconnected_exit:
            parts.append(
                f"exit<{min_exit_roads}: " + ", ".join(underconnected_exit[:8])
            )
        if underconnected_entry:
            parts.append(
                f"entry<{min_entry_roads}: " + ", ".join(underconnected_entry[:8])
            )
        if underconnected_neighbors:
            parts.append(
                f"neighbors<{min_neighbor_districts}: "
                + ", ".join(underconnected_neighbors[:8])
            )
        raise ValueError(
            "District external connectivity too low: " + " | ".join(parts)
        )


def validate_routes(
    flow_entries: list[dict[str, Any]],
    roads_by_id: dict[str, Any],
) -> None:
    if not flow_entries:
        raise ValueError("Scenario flow is empty.")
    for idx, vehicle in enumerate(flow_entries):
        route = vehicle.get("route", [])
        if not route:
            raise ValueError(f"Flow entry {idx} has empty route.")
        for road_id in route:
            if road_id not in roads_by_id:
                raise ValueError(f"Flow entry {idx} references missing road {road_id}.")
        for left, right in zip(route, route[1:]):
            left_end = roads_by_id[left]["endIntersection"]
            right_start = roads_by_id[right]["startIntersection"]
            if left_end != right_start:
                raise ValueError(
                    f"Invalid route transition: {left} -> {right} in entry {idx}."
                )


def build_road_index(roadnet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {road["id"]: road for road in roadnet.get("roads", [])}


def build_roadlink_index(
    roadnet: dict[str, Any],
) -> dict[str, set[tuple[str, str]]]:
    roadlinks_by_intersection: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for intersection in roadnet.get("intersections", []):
        if intersection.get("virtual", False):
            # CityFlow ignores roadLinks on virtual intersections.
            continue
        iid = intersection["id"]
        for road_link in intersection.get("roadLinks", []):
            pair = (road_link["startRoad"], road_link["endRoad"])
            roadlinks_by_intersection[iid].add(pair)
    return roadlinks_by_intersection


def validate_route_with_reasons(
    route: list[str],
    roads_by_id: dict[str, dict[str, Any]],
    roadlinks_by_intersection: dict[str, set[tuple[str, str]]],
) -> list[str]:
    reasons: list[str] = []
    if not route:
        return ["empty_route"]
    if len(route) < 2:
        return ["route_too_short"]

    for rid in route:
        if rid not in roads_by_id:
            reasons.append(f"missing_road:{rid}")
            return reasons

    for left, right in zip(route, route[1:]):
        left_road = roads_by_id[left]
        right_road = roads_by_id[right]
        shared_intersection = left_road["endIntersection"]
        if shared_intersection != right_road["startIntersection"]:
            reasons.append("mismatched_intersection_transition")
            continue
        if (left, right) not in roadlinks_by_intersection.get(shared_intersection, set()):
            reasons.append("missing_roadlink_transition")
    return reasons


def summarize_route_validation(
    flow_entries: list[dict[str, Any]],
    roads_by_id: dict[str, dict[str, Any]],
    roadlinks_by_intersection: dict[str, set[tuple[str, str]]],
) -> dict[str, Any]:
    reason_counter: Counter[str] = Counter()
    total = len(flow_entries)
    valid = 0
    invalid = 0
    for flow in flow_entries:
        reasons = validate_route_with_reasons(
            route=flow.get("route", []),
            roads_by_id=roads_by_id,
            roadlinks_by_intersection=roadlinks_by_intersection,
        )
        if reasons:
            invalid += 1
            reason_counter.update(reasons)
        else:
            valid += 1
    return {
        "total_routes": total,
        "valid_routes": valid,
        "invalid_routes": invalid,
        "top_failure_reasons": reason_counter.most_common(10),
    }


def compute_scenario_diagnostics(
    flow_entries: list[dict[str, Any]],
    city_graph: CityGraph,
    district_data: DistrictData,
) -> dict[str, Any]:
    roads_by_id = build_road_index(city_graph.roadnet)
    assignment = district_data.intersection_to_district
    road_usage: Counter[str] = Counter()
    origin_counter: Counter[str] = Counter()
    destination_counter: Counter[str] = Counter()
    corridor_counter: Counter[str] = Counter()
    external_origin = 0
    external_destination = 0

    for flow in flow_entries:
        route = flow.get("route", [])
        if not route:
            continue
        first = roads_by_id.get(route[0])
        last = roads_by_id.get(route[-1])
        if first:
            origin_key = assignment.get(first["startIntersection"], "external")
            origin_counter[origin_key] += 1
            if origin_key == "external":
                external_origin += 1
        if last:
            destination_key = assignment.get(last["endIntersection"], "external")
            destination_counter[destination_key] += 1
            if destination_key == "external":
                external_destination += 1
        for road_id in route:
            road_usage[road_id] += 1
            road = roads_by_id[road_id]
            ds = assignment.get(road["startIntersection"], "external")
            de = assignment.get(road["endIntersection"], "external")
            if ds != de:
                corridor_counter[f"{ds}->{de}"] += 1

    total_routes = len(flow_entries)
    total_road_traversals = sum(road_usage.values())
    total_roads = len(roads_by_id)
    used_roads = len(road_usage)
    unused_roads = total_roads - used_roads
    boundary_roads = set(district_data.inter_district_roads)
    gateway_roads = set(city_graph.gateway_roads)
    boundary_usage = sum(
        count for road_id, count in road_usage.items() if road_id in boundary_roads
    )
    gateway_usage = sum(
        count for road_id, count in road_usage.items() if road_id in gateway_roads
    )
    top_road_usage = road_usage.most_common(15)
    top_corridors = corridor_counter.most_common(10)
    total_lanes = sum(road.num_lanes for road in city_graph.directed_roads.values())
    demand_per_lane = total_routes / max(1.0, float(total_lanes))
    avg_route_len = total_road_traversals / max(1.0, float(total_routes))
    concentration = (
        sum(count for _, count in top_road_usage[:10]) / max(1.0, float(total_road_traversals))
    )
    boundary_share = boundary_usage / max(1.0, float(total_road_traversals))
    congestion_score = min(
        100.0,
        1.9 * demand_per_lane
        + 2.3 * avg_route_len
        + 46.0 * concentration
        + 34.0 * boundary_share,
    )
    if congestion_score < 30.0:
        congestion_level = "manageable"
    elif congestion_score < 52.0:
        congestion_level = "moderate"
    elif congestion_score < 72.0:
        congestion_level = "heavy"
    else:
        congestion_level = "extreme"

    return {
        "total_vehicles": total_routes,
        "vehicles_by_origin_district": dict(origin_counter),
        "vehicles_by_destination_district": dict(destination_counter),
        "vehicles_from_external": external_origin,
        "vehicles_to_external": external_destination,
        "roads_used": used_roads,
        "unused_roads": unused_roads,
        "boundary_road_usage": boundary_usage,
        "gateway_road_usage": gateway_usage,
        "boundary_road_share": round(boundary_share, 4),
        "top_used_roads": [
            {
                "road_id": road_id,
                "traversals": count,
                "is_arterial": road_id in city_graph.arterial_roads,
                "is_inter_district": road_id in boundary_roads,
                "is_gateway": road_id in gateway_roads,
            }
            for road_id, count in top_road_usage
        ],
        "top_used_corridors": [
            {"corridor": corridor, "traversals": count}
            for corridor, count in top_corridors
        ],
        "estimated_congestion_intensity": {
            "score": round(congestion_score, 3),
            "level": congestion_level,
            "demand_per_lane": round(demand_per_lane, 3),
            "avg_route_length": round(avg_route_len, 3),
            "road_usage_concentration": round(concentration, 4),
        },
    }
