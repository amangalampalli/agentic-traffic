"""City-level orchestration: topology, districts, scenarios, flows, configs, validation."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config_generator import ConfigGenerator
from .district_generator import DistrictGenerator
from .flow_generator import FlowGenerator
from .roadnet_generator import RoadnetGenerator
from .scenario_generator import ScenarioGenerator
from .schemas import DatasetGenerationConfig, TopologyType
from .utils import (
    build_road_index,
    build_roadlink_index,
    clamp,
    compute_scenario_diagnostics,
    ensure_dir,
    summarize_route_validation,
    validate_district_contiguity,
    validate_district_exit_capacity,
    validate_inter_district_connectivity,
    validate_unique_ids,
    write_json,
)


class CityGenerator:
    """Generate one or many synthetic cities with scenario-specific CityFlow files."""

    def __init__(self) -> None:
        self.roadnet_generator = RoadnetGenerator()
        self.district_generator = DistrictGenerator()
        self.scenario_generator = ScenarioGenerator()
        self.flow_generator = FlowGenerator()
        self.config_generator = ConfigGenerator()

    def generate_dataset(self, config: DatasetGenerationConfig) -> None:
        ensure_dir(config.output_dir)
        failures: list[tuple[str, str]] = []
        for idx in range(config.num_cities):
            city_id = f"city_{idx + 1:04d}"
            city_seed = config.seed + idx * 10_003
            try:
                self.generate_city(
                    city_id=city_id,
                    output_dir=config.output_dir / city_id,
                    config=config,
                    city_seed=city_seed,
                )
            except Exception as exc:
                failures.append((city_id, str(exc)))
                if config.fail_fast:
                    raise
        if failures:
            details = "; ".join(
                f"{city}: {message}" for city, message in failures[:5]
            )
            raise RuntimeError(
                f"Dataset generation failed for {len(failures)} city/cities. {details}"
            )

    def generate_city(
        self,
        city_id: str,
        output_dir: Path,
        config: DatasetGenerationConfig,
        city_seed: int,
    ) -> None:
        ensure_dir(output_dir)
        rng = random.Random(city_seed)
        topology_pool: list[TopologyType] = list(config.topologies)
        if not topology_pool:
            raise ValueError("No topology families provided in configuration.")

        attempts_per_topology = 10
        ordered_topologies = topology_pool.copy()
        rng.shuffle(ordered_topologies)
        max_attempts = attempts_per_topology * len(ordered_topologies)
        attempt_count = 0
        city_graph = None
        district_data = None
        last_error: Exception | None = None
        for topology in ordered_topologies:
            for topology_attempt in range(attempts_per_topology):
                attempt_count += 1
                attempt_seed = city_seed + ((attempt_count - 1) * 1009)
                target_intersections = clamp(
                    rng.randint(
                        config.min_districts * config.min_intersections_per_district,
                        config.max_districts * config.max_intersections_per_district,
                    ),
                    low=config.min_districts * config.min_intersections_per_district,
                    high=config.max_districts * config.max_intersections_per_district + 36,
                )
                try:
                    city_graph = self.roadnet_generator.generate(
                        city_id=city_id,
                        seed=attempt_seed,
                        topology=topology,
                        target_intersections=target_intersections,
                        ring_diagonal_keep_prob=config.ring_diagonal_keep_prob,
                        ring_max_diagonal_fraction=config.ring_max_diagonal_fraction,
                    )
                    validate_unique_ids(city_graph)
                    max_districts = max(
                        config.min_districts,
                        min(
                            config.max_districts,
                            max(
                                2,
                                len(
                                    [
                                        nid
                                        for nid in city_graph.intersections
                                        if nid not in city_graph.gateway_intersections
                                    ]
                                )
                                // max(1, config.min_intersections_per_district),
                            ),
                        ),
                    )
                    if topology == "ring_road":
                        max_districts = min(max_districts, max(6, target_intersections // 10))
                        max_districts = max(config.min_districts, max_districts)
                    num_districts = rng.randint(config.min_districts, max_districts)

                    district_data = self.district_generator.generate(
                        city_graph=city_graph,
                        num_districts=num_districts,
                        seed=attempt_seed + 17,
                    )
                    validate_district_contiguity(city_graph, district_data)
                    validate_inter_district_connectivity(district_data)
                    min_exit_roads = 2 if topology == "ring_road" else 3
                    min_entry_roads = 2 if topology == "ring_road" else 3
                    min_neighbor_districts = 1 if topology == "ring_road" else 2
                    validate_district_exit_capacity(
                        district_data=district_data,
                        min_exit_roads=min_exit_roads,
                        min_entry_roads=min_entry_roads,
                        min_neighbor_districts=min_neighbor_districts,
                    )
                    print(
                        f"[INFO] {city_id} attempt={attempt_count} "
                        f"topology={topology} topology_try={topology_attempt + 1}/{attempts_per_topology} "
                        "generated successfully"
                    )
                    break
                except Exception as exc:
                    message = str(exc)
                    print(
                        f"[WARN] {city_id} attempt={attempt_count} "
                        f"topology={topology} topology_try={topology_attempt + 1}/{attempts_per_topology} "
                        f"failed: {message}"
                    )
                    last_error = exc
                    city_graph = None
                    district_data = None
                    continue

            if city_graph is not None and district_data is not None:
                break

        if city_graph is None or district_data is None:
            raise ValueError(
                f"Unable to produce a structurally valid city after {max_attempts} attempts: {last_error}"
            )

        roadnet_path = output_dir / "roadnet.json"
        write_json(roadnet_path, city_graph.roadnet)

        district_map = {
            "intersection_to_district": district_data.intersection_to_district,
            "district_neighbors": district_data.district_neighbors,
            "boundary_intersections": district_data.boundary_intersections,
            "gateway_intersections": sorted(city_graph.gateway_intersections),
            "gateway_roads": sorted(city_graph.gateway_roads),
            "districts": [
                {
                    "id": d.id,
                    "type": d.district_type,
                    "intersections": d.intersections,
                    "neighbors": d.neighbors,
                    "boundary_intersections": d.boundary_intersections,
                    "entry_roads": d.entry_roads,
                    "exit_roads": d.exit_roads,
                }
                for d in district_data.districts.values()
            ],
        }
        write_json(output_dir / "district_map.json", district_map)

        metadata = self._city_metadata(
            city_id=city_id,
            topology=topology,
            city_seed=city_seed,
            city_graph=city_graph,
            district_data=district_data,
            config=config,
        )
        write_json(output_dir / "metadata.json", metadata)

        print(f"[INFO] {city_id} generated: topology={topology}, districts={len(district_data.districts)}")

        scenario_plans = self.scenario_generator.generate(
            city_graph=city_graph,
            district_data=district_data,
            scenario_names=config.scenarios,
            base_seed=city_seed + 1000,
            config=config,
        )
        self._generate_scenarios(
            output_dir=output_dir,
            city_graph=city_graph,
            district_data=district_data,
            scenario_plans=scenario_plans,
            config=config,
            roadnet_path=roadnet_path,
        )

    def _generate_scenarios(
        self,
        output_dir: Path,
        city_graph: Any,
        district_data: Any,
        scenario_plans: dict[str, Any],
        config: DatasetGenerationConfig,
        roadnet_path: Path,
    ) -> None:
        roads_by_id = build_road_index(city_graph.roadnet)
        roadlinks_by_intersection = build_roadlink_index(city_graph.roadnet)
        scenarios_dir = output_dir / "scenarios"
        ensure_dir(scenarios_dir)

        for scenario_name, plan in scenario_plans.items():
            scenario_dir = scenarios_dir / scenario_name
            ensure_dir(scenario_dir)
            flows = self.flow_generator.generate(
                city_graph=city_graph,
                district_data=district_data,
                scenario=plan,
                simulation_steps=config.simulation_steps,
            )
            validation_summary = summarize_route_validation(
                flow_entries=flows,
                roads_by_id=roads_by_id,
                roadlinks_by_intersection=roadlinks_by_intersection,
            )
            if validation_summary["invalid_routes"] > 0:
                reasons = ", ".join(
                    f"{reason}={count}"
                    for reason, count in validation_summary["top_failure_reasons"]
                )
                raise ValueError(
                    f"{scenario_name} contains invalid routes after generation: {reasons}"
                )
            write_json(scenario_dir / "flow.json", flows)
            diagnostics = compute_scenario_diagnostics(
                flow_entries=flows,
                city_graph=city_graph,
                district_data=district_data,
            )

            config_payload = self.config_generator.generate(
                simulation_steps=config.simulation_steps,
                interval=config.interval,
                seed=plan.seed,
                save_replay=config.save_replay,
                roadnet_file=roadnet_path,
                flow_file=scenario_dir / "flow.json",
                scenario_dir=scenario_dir,
            )
            write_json(scenario_dir / "config.json", config_payload)
            write_json(
                scenario_dir / "scenario_metadata.json",
                {
                    "name": scenario_name,
                    "intensity": plan.intensity,
                    "seed": plan.seed,
                    "trip_multiplier": plan.trip_multiplier,
                    "trip_mix": asdict(plan.trip_mix),
                    "blocked_roads": sorted(plan.blocked_roads),
                    "penalized_roads": plan.penalized_roads,
                    "event_district": plan.event_district,
                    "overload_district": plan.overload_district,
                    "diagnostics": diagnostics,
                    "details": plan.metadata,
                },
            )

    def _city_metadata(
        self,
        city_id: str,
        topology: TopologyType,
        city_seed: int,
        city_graph: Any,
        district_data: Any,
        config: DatasetGenerationConfig,
    ) -> dict[str, Any]:
        total_lanes = sum(
            road.num_lanes for road in city_graph.directed_roads.values()
        )
        district_types = {
            did: district.district_type
            for did, district in district_data.districts.items()
        }
        return {
            "city_id": city_id,
            "topology": topology,
            "seed": city_seed,
            "counts": {
                "intersections": len(city_graph.intersections),
                "roads": len(city_graph.directed_roads),
                "lanes": total_lanes,
                "districts": len(district_data.districts),
            },
            "district_types": district_types,
            "district_adjacency_graph": district_data.district_neighbors,
            "inter_district_connector_roads": district_data.inter_district_roads,
            "arterial_roads": sorted(city_graph.arterial_roads),
            "gateway_intersections": sorted(city_graph.gateway_intersections),
            "gateway_connector_roads": sorted(city_graph.gateway_roads),
            "generation_parameters": {
                "min_districts": config.min_districts,
                "max_districts": config.max_districts,
                "min_intersections_per_district": config.min_intersections_per_district,
                "max_intersections_per_district": config.max_intersections_per_district,
                "simulation_steps": config.simulation_steps,
                "interval": config.interval,
                "intensity_levels": config.intensity_levels,
                "global_demand_multiplier": config.global_demand_multiplier,
                "intensity_distribution": config.intensity_distribution,
                "scenario_demand_multipliers": config.scenario_demand_multipliers,
                "ring_diagonal_keep_prob": config.ring_diagonal_keep_prob,
                "ring_max_diagonal_fraction": config.ring_max_diagonal_fraction,
            },
        }
