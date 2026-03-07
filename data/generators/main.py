"""Preconfigured dataset generation harness (no CLI arguments)."""

from __future__ import annotations

import secrets
from pathlib import Path

from .city_generator import CityGenerator
from .schemas import DatasetGenerationConfig


# -----------------------------------------------------------------------------
# Edit these values to control the default batch generation run.
# -----------------------------------------------------------------------------
NUM_CITIES: int = 100
OUTPUT_DIR: Path = Path("data/generated")
BASE_SEED: int | None = 42  # set to an int for reproducible runs
TOPOLOGIES: list[str] = [
    "rectangular_grid",
    "irregular_grid",
    "arterial_local",
    "ring_road",
    "mixed",
]
MIN_DISTRICTS: int = 6
MAX_DISTRICTS: int = 20
MIN_INTERSECTIONS_PER_DISTRICT: int = 4
MAX_INTERSECTIONS_PER_DISTRICT: int = 10
SIMULATION_STEPS: int = 3600
INTERVAL: float = 1.0
FAIL_FAST: bool = False


def main() -> None:
    """Run deterministic-configured dataset generation with pre-set defaults."""
    base_seed = BASE_SEED if BASE_SEED is not None else secrets.randbits(63)

    config = DatasetGenerationConfig(
        num_cities=NUM_CITIES,
        output_dir=OUTPUT_DIR,
        seed=base_seed,
        topologies=TOPOLOGIES,
        min_districts=MIN_DISTRICTS,
        max_districts=MAX_DISTRICTS,
        min_intersections_per_district=MIN_INTERSECTIONS_PER_DISTRICT,
        max_intersections_per_district=MAX_INTERSECTIONS_PER_DISTRICT,
        simulation_steps=SIMULATION_STEPS,
        interval=INTERVAL,
        fail_fast=FAIL_FAST,
    )

    print(f"Generating {config.num_cities} cities into {config.output_dir}")
    print(f"Base seed: {config.seed}")
    print(f"Topologies: {', '.join(TOPOLOGIES)}")

    CityGenerator().generate_dataset(config)


if __name__ == "__main__":
    main()
