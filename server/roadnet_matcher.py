"""Match an uploaded roadnet.json to a known city in the dataset by fingerprint."""
from __future__ import annotations

import json
from pathlib import Path


def match_city_by_roadnet(roadnet_data: dict, generated_root: Path) -> str | None:
    """Return the city_id whose roadnet.json matches the uploaded data, or None."""
    uploaded_fp = _fingerprint(roadnet_data)
    if not uploaded_fp:
        return None

    for city_dir in sorted(generated_root.glob("city_*")):
        roadnet_path = city_dir / "roadnet.json"
        if not roadnet_path.exists():
            continue
        try:
            candidate = json.loads(roadnet_path.read_text())
            if _fingerprint(candidate) == uploaded_fp:
                return city_dir.name
        except Exception:
            continue

    return None


def list_all_cities(generated_root: Path) -> list[str]:
    return sorted(
        d.name
        for d in generated_root.glob("city_*")
        if d.is_dir() and (d / "roadnet.json").exists()
    )


def list_scenarios_for_city(city_id: str, generated_root: Path) -> list[str]:
    scenario_root = generated_root / city_id / "scenarios"
    if not scenario_root.exists():
        return []
    return sorted(
        d.name
        for d in scenario_root.iterdir()
        if d.is_dir()
        and (d / "config.json").exists()
        and (d / "flow.json").exists()
    )


def _fingerprint(roadnet: dict) -> frozenset[str]:
    """Fingerprint = set of non-virtual intersection IDs."""
    return frozenset(
        item["id"]
        for item in roadnet.get("intersections", [])
        if not item.get("virtual", False) and item.get("id")
    )
