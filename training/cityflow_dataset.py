from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SCENARIOS = (
    "normal",
    "morning_rush",
    "evening_rush",
    "accident",
    "construction",
    "district_overload",
    "event_spike",
)


@dataclass(frozen=True)
class ScenarioSpec:
    city_id: str
    scenario_name: str
    city_dir: Path
    scenario_dir: Path
    config_path: Path
    roadnet_path: Path
    district_map_path: Path
    metadata_path: Path


class CityFlowDataset:
    def __init__(
        self,
        generated_root: str | Path = "data/generated",
        splits_root: str | Path = "data/splits",
    ):
        self.generated_root = Path(generated_root)
        self.splits_root = Path(splits_root)

    def discover_cities(self) -> list[str]:
        return sorted(
            city_dir.name
            for city_dir in self.generated_root.glob("city_*")
            if city_dir.is_dir() and (city_dir / "roadnet.json").exists()
        )

    def scenarios_for_city(self, city_id: str) -> list[str]:
        scenario_root = self.generated_root / city_id / "scenarios"
        return sorted(
            scenario_dir.name
            for scenario_dir in scenario_root.iterdir()
            if scenario_dir.is_dir()
            and (scenario_dir / "config.json").exists()
            and (scenario_dir / "flow.json").exists()
        )

    def load_split(self, split_name: str, create_if_missing: bool = True) -> list[str]:
        split_path = self.splits_root / f"{split_name}_cities.txt"
        if not split_path.exists():
            if not create_if_missing:
                raise FileNotFoundError(f"Missing split file: {split_path}")
            self.generate_default_splits()

        split_cities = [
            line.strip()
            for line in split_path.read_text().splitlines()
            if line.strip()
        ]
        available_cities = set(self.discover_cities())
        filtered_cities = [city_id for city_id in split_cities if city_id in available_cities]
        if filtered_cities:
            return filtered_cities

        if create_if_missing and available_cities:
            splits = self.generate_default_splits(overwrite=True)
            return splits.get(split_name, [])

        return filtered_cities

    def generate_default_splits(
        self,
        seed: int = 7,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        overwrite: bool = False,
    ) -> dict[str, list[str]]:
        self.splits_root.mkdir(parents=True, exist_ok=True)

        train_path = self.splits_root / "train_cities.txt"
        val_path = self.splits_root / "val_cities.txt"
        test_path = self.splits_root / "test_cities.txt"
        if not overwrite and train_path.exists() and val_path.exists() and test_path.exists():
            return {
                "train": self.load_split("train", create_if_missing=False),
                "val": self.load_split("val", create_if_missing=False),
                "test": self.load_split("test", create_if_missing=False),
            }

        city_ids = self.discover_cities()
        rng = random.Random(seed)
        rng.shuffle(city_ids)

        num_cities = len(city_ids)
        if num_cities == 0:
            splits = {"train": [], "val": [], "test": []}
        elif num_cities == 1:
            splits = {"train": city_ids[:], "val": city_ids[:], "test": city_ids[:]}
        elif num_cities == 2:
            splits = {
                "train": sorted(city_ids[:1]),
                "val": sorted(city_ids[1:2]),
                "test": sorted(city_ids[:1]),
            }
        else:
            train_count = max(1, int(num_cities * train_ratio))
            val_count = int(num_cities * val_ratio)
            if train_count + val_count >= num_cities:
                val_count = max(0, num_cities - train_count - 1)
            test_count = num_cities - train_count - val_count
            if test_count <= 0:
                test_count = 1
                if val_count > 0:
                    val_count -= 1
                else:
                    train_count = max(1, train_count - 1)

            splits = {
                "train": sorted(city_ids[:train_count]),
                "val": sorted(city_ids[train_count : train_count + val_count]),
                "test": sorted(city_ids[train_count + val_count :]),
            }

        for split_name, city_list in splits.items():
            split_path = self.splits_root / f"{split_name}_cities.txt"
            split_path.write_text("\n".join(city_list) + "\n")

        return splits

    def build_scenario_spec(self, city_id: str, scenario_name: str) -> ScenarioSpec:
        city_dir = self.generated_root / city_id
        scenario_dir = city_dir / "scenarios" / scenario_name
        return ScenarioSpec(
            city_id=city_id,
            scenario_name=scenario_name,
            city_dir=city_dir,
            scenario_dir=scenario_dir,
            config_path=scenario_dir / "config.json",
            roadnet_path=city_dir / "roadnet.json",
            district_map_path=city_dir / "district_map.json",
            metadata_path=city_dir / "metadata.json",
        )

    def sample_scenario(
        self,
        split_name: str,
        rng: random.Random,
        city_id: str | None = None,
        scenario_name: str | None = None,
    ) -> ScenarioSpec:
        available_cities = self.load_split(split_name)
        if not available_cities:
            available_cities = self.discover_cities()
        if not available_cities:
            raise FileNotFoundError(
                f"No generated cities found under {self.generated_root} for split '{split_name}'."
            )
        selected_city = city_id or rng.choice(available_cities)
        selected_scenario = scenario_name or rng.choice(self.scenarios_for_city(selected_city))
        return self.build_scenario_spec(selected_city, selected_scenario)

    def iter_scenarios(
        self,
        split_name: str,
        scenarios_per_city: int | None = None,
        max_cities: int | None = None,
        diversify_single_scenario: bool = False,
    ) -> list[ScenarioSpec]:
        scenario_specs: list[ScenarioSpec] = []
        city_ids = self.load_split(split_name)
        if max_cities is not None:
            city_ids = city_ids[:max_cities]
        for city_index, city_id in enumerate(city_ids):
            scenario_names = self.scenarios_for_city(city_id)
            if (
                diversify_single_scenario
                and scenarios_per_city == 1
                and scenario_names
            ):
                preferred_order = [
                    scenario_name
                    for scenario_name in DEFAULT_SCENARIOS
                    if scenario_name in scenario_names
                ]
                if preferred_order:
                    scenario_names = [preferred_order[city_index % len(preferred_order)]]
                else:
                    scenario_names = [scenario_names[0]]
            elif scenarios_per_city is not None:
                scenario_names = scenario_names[:scenarios_per_city]
            for scenario_name in scenario_names:
                scenario_specs.append(self.build_scenario_spec(city_id, scenario_name))
        return scenario_specs
