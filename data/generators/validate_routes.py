"""Validate generated CityFlow routes and print summary statistics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .utils import build_road_index, build_roadlink_index, summarize_route_validation


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_from_config(config_path: Path) -> tuple[Path, Path]:
    config = _load_json(config_path)
    base_dir = Path(config["dir"])
    roadnet_path = (base_dir / config["roadnetFile"]).resolve()
    flow_path = (base_dir / config["flowFile"]).resolve()
    return roadnet_path, flow_path


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"total routes: {summary['total_routes']}")
    print(f"valid routes: {summary['valid_routes']}")
    print(f"invalid routes: {summary['invalid_routes']}")
    if summary["top_failure_reasons"]:
        formatted = ", ".join(
            f"{reason}={count}" for reason, count in summary["top_failure_reasons"]
        )
        print(f"top failure reasons: {formatted}")
    else:
        print("top failure reasons: none")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CityFlow flow routes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to scenario config.json.",
    )
    parser.add_argument(
        "--roadnet",
        type=Path,
        default=None,
        help="Path to roadnet.json (required if --config not provided).",
    )
    parser.add_argument(
        "--flow",
        type=Path,
        default=None,
        help="Path to flow.json (required if --config not provided).",
    )
    args = parser.parse_args()

    if args.config is not None:
        roadnet_path, flow_path = _resolve_from_config(args.config.resolve())
    else:
        if args.roadnet is None or args.flow is None:
            raise ValueError("Provide --config OR both --roadnet and --flow.")
        roadnet_path = args.roadnet.resolve()
        flow_path = args.flow.resolve()

    if not roadnet_path.exists():
        raise FileNotFoundError(os.fspath(roadnet_path))
    if not flow_path.exists():
        raise FileNotFoundError(os.fspath(flow_path))

    roadnet = _load_json(roadnet_path)
    flows = _load_json(flow_path)
    roads_by_id = build_road_index(roadnet)
    roadlinks_by_intersection = build_roadlink_index(roadnet)
    summary = summarize_route_validation(
        flow_entries=flows,
        roads_by_id=roads_by_id,
        roadlinks_by_intersection=roadlinks_by_intersection,
    )
    _print_summary(summary)
    return 1 if summary["invalid_routes"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
