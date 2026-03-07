"""CLI entrypoint for synthetic CityFlow dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .city_generator import CityGenerator
from .schemas import DatasetGenerationConfig, DemandIntensity

DEFAULT_TOPOLOGIES = ["irregular_grid"]
DEFAULT_SCENARIOS = [
    "normal",
    "morning_rush",
    "evening_rush",
    "accident",
    "construction",
    "event_spike",
    "district_overload",
]
DEFAULT_INTENSITY_LEVELS: list[DemandIntensity] = [
    "normal",
    "moderate_rush",
    "heavy_rush",
    "overload",
    "accident_overload",
]
DEFAULT_INTENSITY_DISTRIBUTION: dict[DemandIntensity, float] = {
    "normal": 0.20,
    "moderate_rush": 0.42,
    "heavy_rush": 0.24,
    "overload": 0.10,
    "accident_overload": 0.04,
}
DEFAULT_SCENARIO_DEMAND_MULTIPLIERS: dict[str, float] = {
    "normal": 1.15,
    "morning_rush": 1.35,
    "evening_rush": 1.35,
    "accident": 1.75,
    "construction": 1.55,
    "event_spike": 1.65,
    "district_overload": 1.70,
}


def _parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",")]
    return [v for v in values if v]


def _parse_key_value_floats(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    result: dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected key=value pair, got '{token}'.")
        key, value = token.split("=", 1)
        result[key.strip()] = float(value.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CityFlow cities with district-aware scenarios."
    )
    parser.add_argument("--num-cities", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-districts", type=int, default=6)
    parser.add_argument("--max-districts", type=int, default=20)
    parser.add_argument("--min-intersections-per-district", type=int, default=4)
    parser.add_argument("--max-intersections-per-district", type=int, default=10)
    parser.add_argument(
        "--topologies",
        type=str,
        default=None,
        help="Comma-separated list of topologies.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated list of scenarios.",
    )
    parser.add_argument("--simulation-steps", type=int, default=3600)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--intensity-levels",
        type=str,
        default=None,
        help="Comma-separated intensity levels.",
    )
    parser.add_argument(
        "--intensity-distribution",
        type=str,
        default=None,
        help="Comma-separated key=value weights for intensities.",
    )
    parser.add_argument(
        "--global-demand-multiplier",
        type=float,
        default=1.25,
        help="Global demand multiplier across all scenarios.",
    )
    parser.add_argument(
        "--scenario-demand-multipliers",
        type=str,
        default=None,
        help="Comma-separated key=value multipliers by scenario name.",
    )
    parser.add_argument(
        "--ring-diagonal-keep-prob",
        type=float,
        default=0.07,
        help="Keep probability for optional ring-road interior diagonals.",
    )
    parser.add_argument(
        "--ring-max-diagonal-fraction",
        type=float,
        default=0.03,
        help="Maximum fraction of optional diagonals retained in ring-road topology.",
    )
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    topologies = _parse_csv_list(args.topologies)
    scenarios = _parse_csv_list(args.scenarios)
    intensity_levels = _parse_csv_list(args.intensity_levels)
    intensity_distribution = _parse_key_value_floats(args.intensity_distribution)
    scenario_demand_multipliers = _parse_key_value_floats(
        args.scenario_demand_multipliers
    )
    config = DatasetGenerationConfig(
        num_cities=args.num_cities,
        output_dir=args.output_dir,
        seed=args.seed,
        min_districts=args.min_districts,
        max_districts=args.max_districts,
        min_intersections_per_district=args.min_intersections_per_district,
        max_intersections_per_district=args.max_intersections_per_district,
        topologies=topologies if topologies is not None else DEFAULT_TOPOLOGIES,
        scenarios=scenarios if scenarios is not None else DEFAULT_SCENARIOS,
        intensity_levels=(
            intensity_levels if intensity_levels is not None else DEFAULT_INTENSITY_LEVELS
        ),
        intensity_distribution=(
            intensity_distribution
            if intensity_distribution
            else DEFAULT_INTENSITY_DISTRIBUTION
        ),
        global_demand_multiplier=args.global_demand_multiplier,
        scenario_demand_multipliers=(
            scenario_demand_multipliers
            if scenario_demand_multipliers
            else DEFAULT_SCENARIO_DEMAND_MULTIPLIERS
        ),
        ring_diagonal_keep_prob=args.ring_diagonal_keep_prob,
        ring_max_diagonal_fraction=args.ring_max_diagonal_fraction,
        simulation_steps=args.simulation_steps,
        interval=args.interval,
        save_replay=args.save_replay,
        fail_fast=args.fail_fast,
    )
    CityGenerator().generate_dataset(config)


if __name__ == "__main__":
    main()
