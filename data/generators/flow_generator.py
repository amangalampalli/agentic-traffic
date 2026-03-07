"""Flow generation with district-aware O-D pressure and turn-feasible routing."""

from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from typing import Any

from .schemas import CityGraph, DistrictData, ScenarioPlan
from .utils import (
    build_road_index,
    build_roadlink_index,
    choose_weighted,
    summarize_route_validation,
    validate_route_with_reasons,
)


class FlowGenerator:
    """Create high-pressure CityFlow flow entries for each scenario."""

    DISTRICT_BASE_PRODUCTION = {
        "residential": 1.70,
        "commercial": 0.85,
        "industrial": 0.95,
        "mixed": 1.15,
    }
    DISTRICT_BASE_ATTRACTION = {
        "residential": 0.90,
        "commercial": 1.85,
        "industrial": 1.55,
        "mixed": 1.25,
    }

    INTENSITY_SCALE = {
        "normal": 1.0,
        "moderate_rush": 1.22,
        "heavy_rush": 1.45,
        "overload": 1.75,
        "accident_overload": 2.0,
    }

    def generate(
        self,
        city_graph: CityGraph,
        district_data: DistrictData,
        scenario: ScenarioPlan,
        simulation_steps: int,
    ) -> list[dict[str, Any]]:
        rng = random.Random(scenario.seed)
        district_ids = sorted(district_data.districts.keys())
        if len(district_ids) < 2:
            raise ValueError("Flow generation requires at least 2 districts.")

        base_density = int(scenario.metadata.get("base_demand_per_intersection", 36))
        trip_total = int(
            len(city_graph.intersections)
            * base_density
            * scenario.trip_multiplier
        )
        trip_total = max(260, min(42000, trip_total))

        routing_state = self._build_routing_state(city_graph, scenario)
        district_features = self._build_district_features(city_graph, district_data)
        connector_counts = self._build_connector_counts(city_graph, district_data)
        production, attraction = self._district_weights(
            city_graph=city_graph,
            district_data=district_data,
            district_features=district_features,
            scenario=scenario,
        )
        gateway_context = self._build_gateway_context(city_graph, district_data)
        external_share = self._external_trip_share(scenario, has_gateways=bool(gateway_context["gateways"]))

        flows: list[dict[str, Any]] = []
        max_global_sampling_attempts = max(5000, trip_total * 8)
        global_attempts = 0
        while len(flows) < trip_total and global_attempts < max_global_sampling_attempts:
            global_attempts += 1
            route = None
            max_attempts = 45
            for _ in range(max_attempts):
                external_mode = self._sample_external_mode(
                    rng=rng,
                    external_share=external_share,
                )
                category = self._sample_trip_category(rng, scenario.trip_mix)
                origin_node: str
                destination_node: str
                origin_district: str
                destination_district: str

                if external_mode == "inbound":
                    gateway = self._sample_gateway_for_inbound(
                        rng=rng,
                        gateway_context=gateway_context,
                        attraction_weights=attraction,
                    )
                    if gateway is None:
                        continue
                    destination_district = self._sample_origin_district(
                        rng=rng,
                        district_ids=district_ids,
                        weights=attraction,
                    )
                    origin_node = gateway
                    destination_node = self._sample_intersection_in_district(
                        rng=rng,
                        district_data=district_data,
                        district_id=destination_district,
                        favor_boundary=False,
                        excluded_nodes=gateway_context["anchor_nodes"],
                    )
                elif external_mode == "outbound":
                    origin_district = self._sample_origin_district(
                        rng=rng,
                        district_ids=district_ids,
                        weights=production,
                    )
                    gateway = self._sample_gateway_for_outbound(
                        rng=rng,
                        origin_district=origin_district,
                        gateway_context=gateway_context,
                        connector_counts=connector_counts,
                    )
                    if gateway is None:
                        continue
                    origin_node = self._sample_intersection_in_district(
                        rng=rng,
                        district_data=district_data,
                        district_id=origin_district,
                        favor_boundary=True,
                        excluded_nodes=gateway_context["anchor_nodes"],
                    )
                    destination_node = gateway
                else:
                    origin_district = self._sample_origin_district(
                        rng=rng,
                        district_ids=district_ids,
                        weights=production,
                    )
                    destination_district = self._sample_destination_district(
                        rng=rng,
                        city_graph=city_graph,
                        origin_district=origin_district,
                        category=category,
                        district_data=district_data,
                        district_features=district_features,
                        district_ids=district_ids,
                        attraction_weights=attraction,
                        connector_counts=connector_counts,
                    )
                    origin_node = self._sample_intersection_in_district(
                        rng=rng,
                        district_data=district_data,
                        district_id=origin_district,
                        favor_boundary=(category != "intra"),
                        excluded_nodes=set(),
                    )
                    destination_node = self._sample_intersection_in_district(
                        rng=rng,
                        district_data=district_data,
                        district_id=destination_district,
                        favor_boundary=(category == "long"),
                        excluded_nodes=set(),
                    )
                if origin_node == destination_node:
                    continue

                route = self._find_turn_feasible_route(
                    start_intersection=origin_node,
                    end_intersection=destination_node,
                    road_lookup=routing_state["road_lookup"],
                    start_roads_by_intersection=routing_state["start_roads_by_intersection"],
                    transitions=routing_state["transitions"],
                    road_cost=routing_state["road_cost"],
                )
                if not route:
                    continue
                reasons = validate_route_with_reasons(
                    route=route,
                    roads_by_id=routing_state["road_lookup"],
                    roadlinks_by_intersection=routing_state["roadlinks_by_intersection"],
                )
                if reasons:
                    route = None
                    continue
                break

            if route:
                start_time = self._sample_departure(rng, scenario, simulation_steps)
                flows.append(
                    {
                        "vehicle": {
                            "length": 5.0,
                            "width": 2.0,
                            "maxPosAcc": 2.2,
                            "maxNegAcc": 4.6,
                            "usualPosAcc": 2.0,
                            "usualNegAcc": 4.2,
                            "minGap": 2.2,
                            "maxSpeed": 13.89,
                            "headwayTime": 1.4,
                        },
                        "route": route,
                        "interval": 1.0,
                        "startTime": start_time,
                        "endTime": start_time,
                    }
                )

        completion_ratio = len(flows) / max(1, trip_total)
        min_completion_ratio = (
            0.55 if scenario.name in {"accident", "construction"} else 0.70
        )
        if completion_ratio < min_completion_ratio:
            raise ValueError(
                f"Scenario {scenario.name} produced too few valid flows "
                f"({len(flows)}/{trip_total}, completion={completion_ratio:.3f})."
            )

        if not flows:
            raise ValueError(f"Scenario {scenario.name} produced no valid flows.")
        summary = summarize_route_validation(
            flow_entries=flows,
            roads_by_id=routing_state["road_lookup"],
            roadlinks_by_intersection=routing_state["roadlinks_by_intersection"],
        )
        if summary["invalid_routes"] > 0:
            reasons = ", ".join(
                f"{reason}={count}" for reason, count in summary["top_failure_reasons"]
            )
            raise ValueError(
                f"Scenario {scenario.name} has invalid routes after regeneration: {reasons}"
            )
        return flows

    def _build_gateway_context(
        self,
        city_graph: CityGraph,
        district_data: DistrictData,
    ) -> dict[str, Any]:
        assignment = district_data.intersection_to_district
        gateways = sorted(city_graph.gateway_intersections)
        anchors_by_gateway: dict[str, str] = {}
        district_by_gateway: dict[str, str] = {}
        gateways_by_district: dict[str, list[str]] = defaultdict(list)

        for gateway in gateways:
            neighbors = [
                n for n in city_graph.adjacency.get(gateway, set()) if n in assignment
            ]
            if not neighbors:
                continue
            anchor = sorted(neighbors)[0]
            district_id = assignment[anchor]
            anchors_by_gateway[gateway] = anchor
            district_by_gateway[gateway] = district_id
            gateways_by_district[district_id].append(gateway)

        return {
            "gateways": sorted(anchors_by_gateway.keys()),
            "anchors_by_gateway": anchors_by_gateway,
            "district_by_gateway": district_by_gateway,
            "gateways_by_district": gateways_by_district,
            "anchor_nodes": set(anchors_by_gateway.values()),
        }

    def _external_trip_share(
        self,
        scenario: ScenarioPlan,
        has_gateways: bool,
    ) -> float:
        if not has_gateways:
            return 0.0
        base = {
            "normal": 0.12,
            "morning_rush": 0.16,
            "evening_rush": 0.16,
            "accident": 0.20,
            "construction": 0.18,
            "event_spike": 0.18,
            "district_overload": 0.19,
        }[scenario.name]
        intensity_boost = {
            "normal": 0.00,
            "moderate_rush": 0.02,
            "heavy_rush": 0.04,
            "overload": 0.07,
            "accident_overload": 0.10,
        }.get(scenario.intensity, 0.0)
        return min(0.42, base + intensity_boost)

    def _sample_external_mode(
        self,
        rng: random.Random,
        external_share: float,
    ) -> str:
        if external_share <= 0.0:
            return "none"
        if rng.random() >= external_share:
            return "none"
        return "inbound" if rng.random() < 0.5 else "outbound"

    def _sample_gateway_for_inbound(
        self,
        rng: random.Random,
        gateway_context: dict[str, Any],
        attraction_weights: dict[str, float],
    ) -> str | None:
        gateways = gateway_context["gateways"]
        if not gateways:
            return None
        district_by_gateway = gateway_context["district_by_gateway"]
        weights: list[float] = []
        for gateway in gateways:
            district_id = district_by_gateway[gateway]
            weights.append(1.0 + attraction_weights.get(district_id, 1.0))
        return choose_weighted(rng, gateways, weights)

    def _sample_gateway_for_outbound(
        self,
        rng: random.Random,
        origin_district: str,
        gateway_context: dict[str, Any],
        connector_counts: dict[tuple[str, str], int],
    ) -> str | None:
        gateways = gateway_context["gateways"]
        if not gateways:
            return None
        district_by_gateway = gateway_context["district_by_gateway"]
        weights: list[float] = []
        for gateway in gateways:
            gateway_district = district_by_gateway[gateway]
            connector_bonus = 1.0 + connector_counts.get(
                (origin_district, gateway_district), 0
            )
            same_district_bonus = 2.0 if gateway_district == origin_district else 1.0
            weights.append(connector_bonus * same_district_bonus)
        return choose_weighted(rng, gateways, weights)

    def _build_routing_state(
        self,
        city_graph: CityGraph,
        scenario: ScenarioPlan,
    ) -> dict[str, Any]:
        road_lookup = build_road_index(city_graph.roadnet)
        roadlinks_by_intersection = build_roadlink_index(city_graph.roadnet)
        start_roads_by_intersection: dict[str, list[str]] = defaultdict(list)
        road_cost: dict[str, float] = {}
        available_roads: set[str] = set()

        for road in city_graph.directed_roads.values():
            if road.id in scenario.blocked_roads:
                continue
            cost = max(1.0, road.length / max(road.speed_limit, 1.0))
            if road.id in scenario.penalized_roads:
                cost *= scenario.penalized_roads[road.id]
            road_cost[road.id] = cost
            available_roads.add(road.id)
            start_roads_by_intersection[road.start_intersection].append(road.id)

        transitions: dict[str, list[str]] = defaultdict(list)
        for pairs in roadlinks_by_intersection.values():
            for start_road, end_road in pairs:
                if start_road not in available_roads or end_road not in available_roads:
                    continue
                transitions[start_road].append(end_road)
        return {
            "road_lookup": road_lookup,
            "roadlinks_by_intersection": roadlinks_by_intersection,
            "start_roads_by_intersection": start_roads_by_intersection,
            "transitions": transitions,
            "road_cost": road_cost,
        }

    def _find_turn_feasible_route(
        self,
        start_intersection: str,
        end_intersection: str,
        road_lookup: dict[str, dict[str, Any]],
        start_roads_by_intersection: dict[str, list[str]],
        transitions: dict[str, list[str]],
        road_cost: dict[str, float],
    ) -> list[str] | None:
        if start_intersection == end_intersection:
            return None
        start_roads = start_roads_by_intersection.get(start_intersection, [])
        if not start_roads:
            return None

        queue: list[tuple[float, str]] = []
        dist: dict[str, float] = {}
        prev: dict[str, str | None] = {}

        for road_id in start_roads:
            if road_id not in road_cost:
                continue
            cost = road_cost[road_id]
            dist[road_id] = cost
            prev[road_id] = None
            heapq.heappush(queue, (cost, road_id))

        best_terminal: str | None = None
        while queue:
            current_cost, current_road = heapq.heappop(queue)
            if current_cost > dist.get(current_road, float("inf")):
                continue

            current_end = road_lookup[current_road]["endIntersection"]
            if current_end == end_intersection:
                best_terminal = current_road
                break

            for next_road in transitions.get(current_road, []):
                next_cost = current_cost + road_cost[next_road]
                if next_cost < dist.get(next_road, float("inf")):
                    dist[next_road] = next_cost
                    prev[next_road] = current_road
                    heapq.heappush(queue, (next_cost, next_road))

        if best_terminal is None:
            return None

        route: list[str] = []
        cursor: str | None = best_terminal
        while cursor is not None:
            route.append(cursor)
            cursor = prev[cursor]
        route.reverse()
        return route

    def _build_district_features(
        self,
        city_graph: CityGraph,
        district_data: DistrictData,
    ) -> dict[str, dict[str, float]]:
        features: dict[str, dict[str, float]] = {}
        for did, district in district_data.districts.items():
            members = district.intersections
            size = len(members)
            cx = sum(city_graph.intersections[n][0] for n in members) / max(1, size)
            cy = sum(city_graph.intersections[n][1] for n in members) / max(1, size)
            features[did] = {
                "size": float(size),
                "neighbors": float(len(district.neighbors)),
                "exits": float(len(district.exit_roads)),
                "boundary": float(len(district.boundary_intersections)),
                "cx": cx,
                "cy": cy,
            }
        return features

    def _build_connector_counts(
        self,
        city_graph: CityGraph,
        district_data: DistrictData,
    ) -> dict[tuple[str, str], int]:
        connector_counts: dict[tuple[str, str], int] = defaultdict(int)
        assignment = district_data.intersection_to_district
        for a, neighbors in city_graph.adjacency.items():
            if a not in assignment:
                continue
            da = assignment[a]
            for b in neighbors:
                if b not in assignment:
                    continue
                db = assignment[b]
                if da == db:
                    continue
                connector_counts[(da, db)] += 1
        return connector_counts

    def _district_weights(
        self,
        city_graph: CityGraph,
        district_data: DistrictData,
        district_features: dict[str, dict[str, float]],
        scenario: ScenarioPlan,
    ) -> tuple[dict[str, float], dict[str, float]]:
        production: dict[str, float] = {}
        attraction: dict[str, float] = {}
        intensity = self.INTENSITY_SCALE.get(scenario.intensity, 1.0)

        for did, district in district_data.districts.items():
            feature = district_features[did]
            size_factor = max(0.85, min(2.1, math.sqrt(feature["size"]) / 2.0))
            connector_factor = 1.0 + min(1.6, feature["exits"] / 7.0)
            base_prod = self.DISTRICT_BASE_PRODUCTION[district.district_type]
            base_attr = self.DISTRICT_BASE_ATTRACTION[district.district_type]
            production[did] = base_prod * size_factor * (0.60 + 0.40 * connector_factor)
            attraction[did] = base_attr * size_factor * (0.62 + 0.38 * connector_factor)

        if scenario.name == "morning_rush":
            self._scale_by_type(district_data, production, "residential", 3.0 * intensity)
            self._scale_by_type(district_data, production, "mixed", 1.3 * intensity)
            self._scale_by_type(district_data, attraction, "commercial", 3.2 * intensity)
            self._scale_by_type(district_data, attraction, "industrial", 2.8 * intensity)
            self._scale_by_type(district_data, attraction, "residential", 0.58)
        elif scenario.name == "evening_rush":
            self._scale_by_type(district_data, production, "commercial", 3.1 * intensity)
            self._scale_by_type(district_data, production, "industrial", 2.7 * intensity)
            self._scale_by_type(district_data, attraction, "residential", 3.0 * intensity)
            self._scale_by_type(district_data, attraction, "commercial", 0.62)
        elif scenario.name == "event_spike" and scenario.event_district:
            attraction[scenario.event_district] *= 3.8 * intensity
            production[scenario.event_district] *= 1.9 * intensity
        elif scenario.name == "district_overload" and scenario.overload_district:
            production[scenario.overload_district] *= 3.2 * intensity
            attraction[scenario.overload_district] *= 3.0 * intensity

        if scenario.name in {"accident", "construction"}:
            impacted = self._impacted_districts(city_graph, district_data, scenario)
            for did in impacted:
                attraction[did] *= 1.6 * intensity
                production[did] *= 1.45 * intensity
                for neighbor in district_data.district_neighbors.get(did, []):
                    attraction[neighbor] *= 1.18
                    production[neighbor] *= 1.16

        return production, attraction

    def _impacted_districts(
        self,
        city_graph: CityGraph,
        district_data: DistrictData,
        scenario: ScenarioPlan,
    ) -> set[str]:
        impacted: set[str] = set()
        assignment = district_data.intersection_to_district
        for road_id in set(scenario.blocked_roads) | set(scenario.penalized_roads.keys()):
            road = city_graph.directed_roads.get(road_id)
            if road is None:
                continue
            if road.start_intersection in assignment:
                impacted.add(assignment[road.start_intersection])
            if road.end_intersection in assignment:
                impacted.add(assignment[road.end_intersection])
        return impacted

    def _scale_by_type(
        self,
        district_data: DistrictData,
        weights: dict[str, float],
        district_type: str,
        factor: float,
    ) -> None:
        for did, district in district_data.districts.items():
            if district.district_type == district_type:
                weights[did] *= factor

    def _sample_trip_category(self, rng: random.Random, trip_mix: Any) -> str:
        labels = ["intra", "adjacent", "long"]
        weights = [
            trip_mix.intra_district,
            trip_mix.adjacent_district,
            trip_mix.long_distance,
        ]
        return choose_weighted(rng, labels, weights)

    def _sample_origin_district(
        self,
        rng: random.Random,
        district_ids: list[str],
        weights: dict[str, float],
    ) -> str:
        values = district_ids
        scalar = [weights[d] for d in district_ids]
        return choose_weighted(rng, values, scalar)

    def _sample_destination_district(
        self,
        rng: random.Random,
        city_graph: CityGraph,
        origin_district: str,
        category: str,
        district_data: DistrictData,
        district_features: dict[str, dict[str, float]],
        district_ids: list[str],
        attraction_weights: dict[str, float],
        connector_counts: dict[tuple[str, str], int],
    ) -> str:
        if category == "intra":
            return origin_district

        if category == "adjacent":
            neighbors = district_data.district_neighbors.get(origin_district, [])
            if neighbors:
                weights = []
                for neighbor in neighbors:
                    connector = 1.0 + connector_counts.get((origin_district, neighbor), 0)
                    weights.append(attraction_weights[neighbor] * connector)
                return choose_weighted(rng, neighbors, weights)

        origin_feature = district_features[origin_district]
        candidates = [d for d in district_ids if d != origin_district]
        if category == "long":
            candidates = [
                did
                for did in candidates
                if did not in district_data.district_neighbors.get(origin_district, [])
            ] or [d for d in district_ids if d != origin_district]

        weights: list[float] = []
        for candidate in candidates:
            feature = district_features[candidate]
            dx = feature["cx"] - origin_feature["cx"]
            dy = feature["cy"] - origin_feature["cy"]
            distance = math.hypot(dx, dy)
            normalized_distance = max(1.0, distance / 260.0)
            corridor_bonus = 1.0 + min(
                1.5,
                (
                    feature["exits"] + feature["neighbors"]
                ) / 10.0,
            )
            if category == "long":
                weight = attraction_weights[candidate] * normalized_distance * corridor_bonus
            else:
                weight = attraction_weights[candidate] * (0.85 + 0.15 * corridor_bonus)
            weights.append(weight)
        return choose_weighted(rng, candidates, weights)

    def _sample_intersection_in_district(
        self,
        rng: random.Random,
        district_data: DistrictData,
        district_id: str,
        favor_boundary: bool,
        excluded_nodes: set[str],
    ) -> str:
        district = district_data.districts[district_id]
        values = [node for node in district.intersections if node not in excluded_nodes]
        if not values:
            values = district.intersections
        boundary = set(district.boundary_intersections)
        weights: list[float] = []
        for node in values:
            if node in boundary:
                weights.append(2.3 if favor_boundary else 1.15)
            else:
                weights.append(0.95 if favor_boundary else 1.2)
        return choose_weighted(rng, values, weights)

    def _sample_departure(
        self,
        rng: random.Random,
        scenario: ScenarioPlan,
        simulation_steps: int,
    ) -> int:
        windows = scenario.departure_windows
        labels = list(range(len(windows)))
        weights = [w for _, _, w in windows]
        selected = choose_weighted(rng, [str(i) for i in labels], weights)
        window = windows[int(selected)]
        start = int(window[0] * simulation_steps)
        end = int(window[1] * simulation_steps)
        if end <= start:
            end = start + 1
        return rng.randint(start, max(start, end - 1))
