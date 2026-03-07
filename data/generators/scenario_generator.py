"""Scenario plan generation for demand intensity and network disturbances."""

from __future__ import annotations

import random
from collections import defaultdict

from .schemas import (
    CityGraph,
    DatasetGenerationConfig,
    DemandIntensity,
    DistrictData,
    ScenarioPlan,
    ScenarioType,
    TripMix,
)


class ScenarioGenerator:
    """Generate scenario-specific modifiers, bottlenecks, and demand intensity."""

    INTENSITY_MULTIPLIER: dict[DemandIntensity, float] = {
        "normal": 1.0,
        "moderate_rush": 1.45,
        "heavy_rush": 2.1,
        "overload": 2.9,
        "accident_overload": 3.6,
    }

    SCENARIO_BASE_MULTIPLIER: dict[ScenarioType, float] = {
        "normal": 1.15,
        "morning_rush": 1.35,
        "evening_rush": 1.35,
        "accident": 1.85,
        "construction": 1.65,
        "event_spike": 1.75,
        "district_overload": 1.80,
    }

    BASE_DEMAND_PER_INTERSECTION: dict[ScenarioType, int] = {
        "normal": 42,
        "morning_rush": 52,
        "evening_rush": 52,
        "accident": 60,
        "construction": 56,
        "event_spike": 62,
        "district_overload": 64,
    }

    INTENSITY_ALLOWED_BY_SCENARIO: dict[ScenarioType, list[DemandIntensity]] = {
        "normal": ["normal", "moderate_rush", "heavy_rush"],
        "morning_rush": ["moderate_rush", "heavy_rush", "overload"],
        "evening_rush": ["moderate_rush", "heavy_rush", "overload"],
        "accident": ["heavy_rush", "overload", "accident_overload"],
        "construction": ["moderate_rush", "heavy_rush", "overload", "accident_overload"],
        "event_spike": ["moderate_rush", "heavy_rush", "overload", "accident_overload"],
        "district_overload": ["moderate_rush", "heavy_rush", "overload", "accident_overload"],
    }

    def generate(
        self,
        city_graph: CityGraph,
        district_data: DistrictData,
        scenario_names: list[ScenarioType],
        base_seed: int,
        config: DatasetGenerationConfig,
    ) -> dict[str, ScenarioPlan]:
        plans: dict[str, ScenarioPlan] = {}
        for idx, name in enumerate(scenario_names):
            seed = base_seed + (idx * 101)
            rng = random.Random(seed)
            intensity = self._sample_intensity(name, rng, config)
            trip_multiplier = self._trip_multiplier(name, intensity, config)
            trip_mix = self._trip_mix(name, intensity)
            departure_windows = self._departure_windows(name, intensity)
            base_demand = self.BASE_DEMAND_PER_INTERSECTION[name]

            if name == "normal":
                plans[name] = ScenarioPlan(
                    name=name,
                    intensity=intensity,
                    seed=seed,
                    trip_multiplier=trip_multiplier,
                    trip_mix=trip_mix,
                    departure_windows=departure_windows,
                    metadata={
                        "description": "Balanced baseline traffic with sampled intensity.",
                        "base_demand_per_intersection": base_demand,
                    },
                )
            elif name == "morning_rush":
                plans[name] = ScenarioPlan(
                    name=name,
                    intensity=intensity,
                    seed=seed,
                    trip_multiplier=trip_multiplier,
                    trip_mix=trip_mix,
                    departure_windows=departure_windows,
                    metadata={
                        "description": "Strong residential outbound and work-district inbound pressure.",
                        "base_demand_per_intersection": base_demand,
                    },
                )
            elif name == "evening_rush":
                plans[name] = ScenarioPlan(
                    name=name,
                    intensity=intensity,
                    seed=seed,
                    trip_multiplier=trip_multiplier,
                    trip_mix=trip_mix,
                    departure_windows=departure_windows,
                    metadata={
                        "description": "Strong work-district outbound and residential inbound pressure.",
                        "base_demand_per_intersection": base_demand,
                    },
                )
            elif name == "accident":
                blocked, penalized = self._accident_impairments(
                    city_graph, district_data, intensity, rng
                )
                plans[name] = ScenarioPlan(
                    name=name,
                    intensity=intensity,
                    seed=seed,
                    trip_multiplier=trip_multiplier,
                    trip_mix=trip_mix,
                    departure_windows=departure_windows,
                    blocked_roads=blocked,
                    penalized_roads=penalized,
                    metadata={
                        "description": "Severe disruption on connector/arterial corridors.",
                        "base_demand_per_intersection": base_demand,
                        "accident_roads": sorted(blocked),
                        "bottleneck_penalties": {k: v for k, v in penalized.items() if k not in blocked},
                    },
                )
            elif name == "construction":
                blocked, penalized = self._construction_impairments(
                    city_graph, district_data, intensity, rng
                )
                plans[name] = ScenarioPlan(
                    name=name,
                    intensity=intensity,
                    seed=seed,
                    trip_multiplier=trip_multiplier,
                    trip_mix=trip_mix,
                    departure_windows=departure_windows,
                    blocked_roads=blocked,
                    penalized_roads=penalized,
                    metadata={
                        "description": "Localized but severe construction bottlenecks.",
                        "base_demand_per_intersection": base_demand,
                        "construction_roads": sorted(blocked),
                        "bottleneck_penalties": {k: v for k, v in penalized.items() if k not in blocked},
                    },
                )
            elif name == "event_spike":
                event_district = self._pick_high_pressure_district(district_data, rng)
                plans[name] = ScenarioPlan(
                    name=name,
                    intensity=intensity,
                    seed=seed,
                    trip_multiplier=trip_multiplier,
                    trip_mix=trip_mix,
                    departure_windows=departure_windows,
                    event_district=event_district,
                    metadata={
                        "description": "Pre-event surge and outbound release around event district.",
                        "base_demand_per_intersection": base_demand,
                        "event_district": event_district,
                    },
                )
            elif name == "district_overload":
                overload = self._pick_high_pressure_district(district_data, rng)
                plans[name] = ScenarioPlan(
                    name=name,
                    intensity=intensity,
                    seed=seed,
                    trip_multiplier=trip_multiplier,
                    trip_mix=trip_mix,
                    departure_windows=departure_windows,
                    overload_district=overload,
                    metadata={
                        "description": "One district receives amplified production and attraction.",
                        "base_demand_per_intersection": base_demand,
                        "overload_district": overload,
                    },
                )
            else:
                raise ValueError(f"Unsupported scenario: {name}")
        return plans

    def _sample_intensity(
        self,
        scenario: ScenarioType,
        rng: random.Random,
        config: DatasetGenerationConfig,
    ) -> DemandIntensity:
        allowed = self.INTENSITY_ALLOWED_BY_SCENARIO[scenario]
        candidates = [i for i in config.intensity_levels if i in allowed]
        if not candidates:
            candidates = allowed
        scenario_bias: dict[ScenarioType, dict[DemandIntensity, float]] = {
            "normal": {"normal": 1.35},
            "morning_rush": {"heavy_rush": 1.4, "overload": 1.45},
            "evening_rush": {"heavy_rush": 1.4, "overload": 1.45},
            "accident": {"overload": 1.8, "accident_overload": 2.2},
            "construction": {"heavy_rush": 1.3, "overload": 1.7, "accident_overload": 2.0},
            "event_spike": {"heavy_rush": 1.35, "overload": 1.65, "accident_overload": 1.8},
            "district_overload": {"heavy_rush": 1.35, "overload": 1.75, "accident_overload": 1.95},
        }
        bias = scenario_bias.get(scenario, {})
        weights = [
            max(0.0, config.intensity_distribution.get(i, 0.0)) * bias.get(i, 1.0)
            for i in candidates
        ]
        if sum(weights) <= 0.0:
            weights = [1.0] * len(candidates)
        return rng.choices(candidates, weights=weights, k=1)[0]

    def _trip_multiplier(
        self,
        scenario: ScenarioType,
        intensity: DemandIntensity,
        config: DatasetGenerationConfig,
    ) -> float:
        base = self.SCENARIO_BASE_MULTIPLIER[scenario]
        intensity_scale = self.INTENSITY_MULTIPLIER[intensity]
        per_scenario = config.scenario_demand_multipliers.get(scenario, 1.0)
        return base * intensity_scale * config.global_demand_multiplier * per_scenario

    def _trip_mix(
        self,
        scenario: ScenarioType,
        intensity: DemandIntensity,
    ) -> TripMix:
        base: dict[ScenarioType, tuple[float, float, float]] = {
            "normal": (0.44, 0.34, 0.22),
            "morning_rush": (0.34, 0.38, 0.28),
            "evening_rush": (0.34, 0.38, 0.28),
            "accident": (0.28, 0.40, 0.32),
            "construction": (0.30, 0.40, 0.30),
            "event_spike": (0.24, 0.40, 0.36),
            "district_overload": (0.26, 0.42, 0.32),
        }
        intra, adjacent, long = base[scenario]
        intensity_shift = {
            "normal": 0.00,
            "moderate_rush": 0.03,
            "heavy_rush": 0.06,
            "overload": 0.09,
            "accident_overload": 0.12,
        }[intensity]
        intra = max(0.14, intra - intensity_shift)
        adjacent = min(0.56, adjacent + (0.55 * intensity_shift))
        long = min(0.44, long + (0.45 * intensity_shift))
        norm = intra + adjacent + long
        return TripMix(
            intra_district=intra / norm,
            adjacent_district=adjacent / norm,
            long_distance=long / norm,
        )

    def _departure_windows(
        self,
        scenario: ScenarioType,
        intensity: DemandIntensity,
    ) -> list[tuple[float, float, float]]:
        compression = {
            "normal": 1.00,
            "moderate_rush": 0.82,
            "heavy_rush": 0.62,
            "overload": 0.46,
            "accident_overload": 0.35,
        }[intensity]

        if scenario == "morning_rush":
            peak_width = 0.34 * compression
            peak_start = max(0.07, 0.26 - peak_width / 2.0)
            peak_end = min(0.58, peak_start + peak_width)
            return [(0.0, peak_start, 0.12), (peak_start, peak_end, 0.76), (peak_end, 1.0, 0.12)]
        if scenario == "evening_rush":
            peak_width = 0.34 * compression
            peak_end = min(0.95, 0.74 + peak_width / 2.0)
            peak_start = max(0.34, peak_end - peak_width)
            return [(0.0, peak_start, 0.10), (peak_start, peak_end, 0.76), (peak_end, 1.0, 0.14)]
        if scenario in {"event_spike", "district_overload"}:
            peak_width = 0.42 * compression
            peak_start = max(0.20, 0.52 - peak_width / 2.0)
            peak_end = min(0.90, peak_start + peak_width)
            return [(0.0, peak_start, 0.10), (peak_start, peak_end, 0.74), (peak_end, 1.0, 0.16)]
        if scenario in {"accident", "construction"}:
            peak_width = 0.45 * compression
            peak_start = max(0.16, 0.48 - peak_width / 2.0)
            peak_end = min(0.88, peak_start + peak_width)
            return [(0.0, peak_start, 0.16), (peak_start, peak_end, 0.68), (peak_end, 1.0, 0.16)]
        return [(0.0, 0.28, 0.22), (0.28, 0.72, 0.56), (0.72, 1.0, 0.22)]

    def _road_importance_scores(
        self,
        city_graph: CityGraph,
        district_data: DistrictData,
    ) -> dict[str, float]:
        boundary_nodes = set(district_data.boundary_intersections)
        scores: dict[str, float] = {}
        for road_id, road in city_graph.directed_roads.items():
            score = 1.0
            if road_id in city_graph.arterial_roads:
                score += 3.0
            if road_id in district_data.inter_district_roads:
                score += 2.5
            if road.start_intersection in boundary_nodes or road.end_intersection in boundary_nodes:
                score += 1.8
            score += 0.30 * len(city_graph.adjacency[road.start_intersection])
            score += 0.30 * len(city_graph.adjacency[road.end_intersection])
            scores[road_id] = score
        return scores

    def _weighted_sample_without_replacement(
        self,
        candidates: list[str],
        weights: dict[str, float],
        k: int,
        rng: random.Random,
    ) -> list[str]:
        remaining = list(candidates)
        picked: list[str] = []
        k = min(k, len(remaining))
        while remaining and len(picked) < k:
            probs = [max(0.01, weights.get(c, 0.01)) for c in remaining]
            choice = rng.choices(remaining, weights=probs, k=1)[0]
            picked.append(choice)
            remaining.remove(choice)
        return picked

    def _accident_impairments(
        self,
        city_graph: CityGraph,
        district_data: DistrictData,
        intensity: DemandIntensity,
        rng: random.Random,
    ) -> tuple[set[str], dict[str, float]]:
        importance = self._road_importance_scores(city_graph, district_data)
        ranked = sorted(importance.keys(), key=lambda rid: importance[rid], reverse=True)
        block_count = {
            "normal": 1,
            "moderate_rush": 2,
            "heavy_rush": 2,
            "overload": 3,
            "accident_overload": 4,
        }[intensity]
        blocked = set(self._weighted_sample_without_replacement(ranked[:120], importance, block_count, rng))
        severity = {
            "normal": 7.0,
            "moderate_rush": 8.5,
            "heavy_rush": 10.5,
            "overload": 12.5,
            "accident_overload": 14.5,
        }[intensity]
        penalized: dict[str, float] = {rid: severity for rid in blocked}

        # Expand impairment to adjacent connector roads to create spillback.
        by_intersection: dict[str, set[str]] = defaultdict(set)
        for road_id, road in city_graph.directed_roads.items():
            by_intersection[road.start_intersection].add(road_id)
            by_intersection[road.end_intersection].add(road_id)
        for rid in blocked:
            road = city_graph.directed_roads[rid]
            nearby = by_intersection[road.start_intersection] | by_intersection[road.end_intersection]
            for neighbor in nearby:
                if neighbor in blocked:
                    continue
                if neighbor in district_data.inter_district_roads or neighbor in city_graph.arterial_roads:
                    penalized[neighbor] = max(penalized.get(neighbor, 0.0), severity * 0.78)
        return blocked, penalized

    def _construction_impairments(
        self,
        city_graph: CityGraph,
        district_data: DistrictData,
        intensity: DemandIntensity,
        rng: random.Random,
    ) -> tuple[set[str], dict[str, float]]:
        candidate_district = self._pick_high_pressure_district(district_data, rng)
        members = set(district_data.districts[candidate_district].intersections)
        localized = [
            rid
            for rid, road in city_graph.directed_roads.items()
            if road.start_intersection in members or road.end_intersection in members
        ]
        if not localized:
            localized = sorted(city_graph.directed_roads.keys())

        importance = self._road_importance_scores(city_graph, district_data)
        block_count = {
            "normal": 1,
            "moderate_rush": 1,
            "heavy_rush": 2,
            "overload": 3,
            "accident_overload": 4,
        }[intensity]
        penalize_count = {
            "normal": 10,
            "moderate_rush": 16,
            "heavy_rush": 22,
            "overload": 30,
            "accident_overload": 36,
        }[intensity]
        blocked = set(
            self._weighted_sample_without_replacement(localized, importance, block_count, rng)
        )
        penalize_candidates = self._weighted_sample_without_replacement(
            localized, importance, penalize_count, rng
        )
        severity = {
            "normal": 5.0,
            "moderate_rush": 6.5,
            "heavy_rush": 8.5,
            "overload": 10.5,
            "accident_overload": 12.5,
        }[intensity]
        penalized: dict[str, float] = {}
        for rid in penalize_candidates:
            factor = severity
            if rid in blocked:
                factor = severity * 1.35
            penalized[rid] = max(penalized.get(rid, 0.0), factor)
        for rid in blocked:
            penalized[rid] = max(penalized.get(rid, 0.0), severity * 1.55)
        return blocked, penalized

    def _pick_high_pressure_district(
        self,
        district_data: DistrictData,
        rng: random.Random,
    ) -> str:
        district_ids = sorted(district_data.districts.keys())
        weights: list[float] = []
        for did in district_ids:
            district = district_data.districts[did]
            w = 1.0 + 0.20 * len(district.neighbors) + 0.06 * len(district.boundary_intersections)
            weights.append(w)
        return rng.choices(district_ids, weights=weights, k=1)[0]
