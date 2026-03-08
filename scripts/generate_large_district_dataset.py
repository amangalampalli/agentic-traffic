from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from district_llm.generate_dataset import build_env, generate_examples_for_episode
from district_llm.prompting import build_system_prompt
from district_llm.schema import DistrictAction
from district_llm.teachers import BaseTeacher, build_teacher, parse_teacher_spec
from env.observation_builder import ObservationConfig
from env.reward import RewardConfig
from env.traffic_env import EnvConfig
from training.cityflow_dataset import CityFlowDataset, DEFAULT_SCENARIOS

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_OUTPUT_DIR = "data/district_llm_dataset_v3"
SCHEMA_VERSION = "district_action_v1_messages_v3_candidates"


@dataclass(frozen=True)
class CollectionResult:
    rows: list[dict[str, Any]]
    duplicate_rows_removed: int
    low_signal_rows_removed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a large, candidate-constrained district-LLM chat dataset."
    )
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-val", type=int, default=2500)
    parser.add_argument("--cities", type=int, default=12)
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Either 'all' or a comma-separated list such as normal,morning_rush,accident.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument(
        "--teacher-spec",
        action="append",
        default=[],
        help="Repeatable teacher source, e.g. rl_checkpoint=artifacts/dqn_shared/best_validation.pt.",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/dqn_shared/best_validation.pt",
        help="Default DQN checkpoint used when no --teacher-spec is provided.",
    )
    parser.add_argument("--include-non-dqn-sources", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--top-k-congested", type=int, default=3)
    parser.add_argument("--max-candidate-intersections", type=int, default=6)
    parser.add_argument("--max-target-intersections", type=int, default=3)
    parser.add_argument("--fixed-green-time", type=int, default=20)
    return parser.parse_args()


def default_env_config() -> EnvConfig:
    return EnvConfig(
        simulator_interval=1,
        decision_interval=5,
        min_green_time=10,
        thread_num=1,
        max_episode_seconds=None,
        observation=ObservationConfig(),
        reward=RewardConfig(variant="wait_queue_throughput"),
    )


def resolve_teachers(args: argparse.Namespace) -> list[BaseTeacher]:
    teacher_specs = list(args.teacher_spec)
    if not teacher_specs:
        checkpoint_path = Path(args.checkpoint)
        teacher_specs = (
            [f"rl_checkpoint={checkpoint_path}"]
            if checkpoint_path.exists()
            else ["queue_greedy"]
        )

    teachers: list[BaseTeacher] = []
    for spec in teacher_specs:
        controller_type, checkpoint = parse_teacher_spec(spec)
        teacher = build_teacher(
            controller_type=controller_type,
            checkpoint=checkpoint,
            fixed_green_time=args.fixed_green_time,
            seed=args.seed,
            device=args.device,
        )
        if teacher.metadata.controller_family != "dqn" and not args.include_non_dqn_sources:
            continue
        teachers.append(teacher)

    if not teachers:
        raise ValueError("No usable teachers resolved. Provide a DQN checkpoint or enable non-DQN sources.")

    return sorted(
        teachers,
        key=lambda teacher: (teacher.metadata.controller_family != "dqn", teacher.metadata.controller_type),
    )


def parse_scenarios(raw_value: str) -> set[str] | None:
    if raw_value.strip().lower() == "all":
        return None
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def build_balanced_scenario_specs(
    dataset: CityFlowDataset,
    split_name: str,
    max_cities: int,
    allowed_scenarios: set[str] | None,
) -> list[Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    city_ids = dataset.load_split(split_name)[:max_cities]
    scenario_order = {name: index for index, name in enumerate(DEFAULT_SCENARIOS)}

    for city_id in city_ids:
        for scenario_name in dataset.scenarios_for_city(city_id):
            if allowed_scenarios is not None and scenario_name not in allowed_scenarios:
                continue
            grouped[scenario_name].append(dataset.build_scenario_spec(city_id, scenario_name))

    if not grouped:
        raise ValueError(f"No scenario specs found for split={split_name}.")

    for specs in grouped.values():
        specs.sort(key=lambda spec: spec.city_id)

    scenario_names = sorted(
        grouped,
        key=lambda name: (scenario_order.get(name, len(DEFAULT_SCENARIOS)), name),
    )
    interleaved: list[Any] = []
    pending = True
    round_index = 0
    while pending:
        pending = False
        for scenario_name in scenario_names:
            specs = grouped[scenario_name]
            if round_index < len(specs):
                interleaved.append(specs[round_index])
                pending = True
        round_index += 1
    return interleaved


def extract_summary_text(prompt: str) -> str:
    state_marker = "### DISTRICT STATE\n"
    decision_marker = "\n\n### DECISION"
    if state_marker not in prompt:
        return prompt.strip()
    summary_block = prompt.split(state_marker, 1)[1]
    summary_block = summary_block.split(decision_marker, 1)[0].strip()
    return f"### DISTRICT STATE\n{summary_block}"


def is_informative_example(example: dict[str, Any]) -> bool:
    state = example["state"]
    top_congested = state.get("top_congested_intersections", [])
    all_zero_top = all(
        float(item.get("queue_total", 0.0)) <= 0.0
        and float(item.get("wait_total", 0.0)) <= 0.0
        and float(item.get("outgoing_load", 0.0)) <= 0.0
        for item in top_congested
    )
    return not (
        int(state.get("decision_step", 0)) == 0
        or (
            float(state.get("total_queue", 0.0)) <= 0.0
            and float(state.get("total_wait", 0.0)) <= 0.0
            and float(state.get("recent_throughput", 0.0)) <= 0.0
            and not bool(state.get("spillback_risk", False))
            and all_zero_top
        )
    )


def make_message_row(example: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    assistant_content = json.dumps(example["response_json"], sort_keys=True)
    row = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": extract_summary_text(example["prompt"])},
            {"role": "assistant", "content": assistant_content},
        ],
        "city_id": example["city_id"],
        "district_id": example["district_id"],
        "district_type": example["district_type"],
        "scenario": example["scenario"],
        "controller_type": example["controller_type"],
        "controller_family": example["controller_family"],
        "teacher_algorithm": example["teacher_algorithm"],
        "controller_id": example["controller_id"],
        "checkpoint_path": example["checkpoint_path"],
        "candidate_intersections": example.get("candidate_intersections", []),
    }
    validate_row(row)
    return row


def validate_row(row: dict[str, Any]) -> None:
    messages = row.get("messages", [])
    if len(messages) != 3:
        raise ValueError("Each row must contain exactly 3 chat messages.")
    if not messages[1]["content"].strip():
        raise ValueError("User summary content is empty.")
    payload = json.loads(messages[2]["content"])
    action = DistrictAction.from_dict(payload)
    visible_candidates = {
        str(item.get("intersection_id"))
        for item in row.get("candidate_intersections", [])
        if str(item.get("intersection_id", "")).strip()
    }
    if visible_candidates and any(item not in visible_candidates for item in action.target_intersections):
        raise ValueError("target_intersections must remain inside candidate_intersections.")


def row_key(row: dict[str, Any]) -> str:
    return json.dumps(row["messages"], sort_keys=True, separators=(",", ":"))


def collect_split_rows(
    split_name: str,
    target_rows: int,
    scenario_specs: list[Any],
    teachers: list[BaseTeacher],
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> CollectionResult:
    rows: list[dict[str, Any]] = []
    system_prompt = build_system_prompt(
        max_target_intersections=args.max_target_intersections,
        allow_only_visible_candidates=True,
    )
    seen_keys: set[str] = set()
    duplicate_rows_removed = 0
    low_signal_rows_removed = 0
    episode_index = 0
    max_rounds = max(target_rows * 3, len(scenario_specs) * 40)
    progress = (
        tqdm(total=target_rows, desc=f"{split_name} rows", dynamic_ncols=True)
        if tqdm is not None
        else None
    )

    try:
        while len(rows) < target_rows and episode_index < max_rounds:
            scenario_spec = scenario_specs[episode_index % len(scenario_specs)]
            for teacher in teachers:
                if progress is not None:
                    progress.set_postfix_str(
                        " ".join(
                            [
                                f"episode={episode_index}",
                                f"city={scenario_spec.city_id}",
                                f"scenario={scenario_spec.scenario_name}",
                                f"teacher={teacher.metadata.controller_type}",
                            ]
                        )
                    )
                env = build_env(env_config=env_config, scenario_spec=scenario_spec)
                examples = generate_examples_for_episode(
                    env=env,
                    teacher=teacher,
                    district_interval=args.decision_interval,
                    top_k_congested=args.top_k_congested,
                    max_candidate_intersections=args.max_candidate_intersections,
                    max_target_intersections=args.max_target_intersections,
                    episode_index=episode_index,
                )
                for example in examples:
                    if not is_informative_example(example):
                        low_signal_rows_removed += 1
                        continue
                    row = make_message_row(example, system_prompt=system_prompt)
                    key = row_key(row)
                    if key in seen_keys:
                        duplicate_rows_removed += 1
                        continue
                    seen_keys.add(key)
                    rows.append(row)
                    if progress is not None:
                        progress.update(1)
                    if len(rows) >= target_rows:
                        break
                if len(rows) >= target_rows:
                    break
            episode_index += 1
    finally:
        if progress is not None:
            progress.close()

    if len(rows) < target_rows:
        raise RuntimeError(
            f"Unable to collect {target_rows} rows for split={split_name}. "
            f"Collected {len(rows)} rows after filtering."
        )

    return CollectionResult(
        rows=rows[:target_rows],
        duplicate_rows_removed=duplicate_rows_removed,
        low_signal_rows_removed=low_signal_rows_removed,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def counter_dict(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def average(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def build_split_diagnostics(result: CollectionResult) -> dict[str, Any]:
    rows = result.rows
    assistant_payloads = [json.loads(row["messages"][2]["content"]) for row in rows]
    summary_lengths = [len(row["messages"][1]["content"]) for row in rows]
    assistant_lengths = [len(row["messages"][2]["content"]) for row in rows]
    candidate_counts = [len(row.get("candidate_intersections", [])) for row in rows]
    target_counts = [len(payload.get("target_intersections", [])) for payload in assistant_payloads]
    assistant_uniqueness = (
        len({row["messages"][2]["content"] for row in rows}) / len(rows)
        if rows
        else 0.0
    )
    return {
        "rows": len(rows),
        "rows_per_scenario": counter_dict([row["scenario"] for row in rows]),
        "rows_per_district_type": counter_dict([row["district_type"] for row in rows]),
        "rows_per_city": counter_dict([row["city_id"] for row in rows]),
        "controller_family_counts": counter_dict([row["controller_family"] for row in rows]),
        "summary_length": {
            "average": average(summary_lengths),
            "median": median(summary_lengths),
        },
        "assistant_json_length": {
            "average": average(assistant_lengths),
            "median": median(assistant_lengths),
        },
        "candidate_pool_size": {
            "average": average(candidate_counts),
            "median": median(candidate_counts),
        },
        "target_intersections_count": {
            "average": average(target_counts),
            "median": median(target_counts),
        },
        "strategy_distribution": counter_dict([payload["strategy"] for payload in assistant_payloads]),
        "priority_corridor_distribution": counter_dict(
            [str(payload.get("priority_corridor")) for payload in assistant_payloads]
        ),
        "phase_bias_distribution": counter_dict([payload["phase_bias"] for payload in assistant_payloads]),
        "duration_steps_distribution": counter_dict(
            [str(payload["duration_steps"]) for payload in assistant_payloads]
        ),
        "assistant_uniqueness_ratio": assistant_uniqueness,
        "duplicate_rows_removed": result.duplicate_rows_removed,
        "low_signal_rows_removed": result.low_signal_rows_removed,
    }


def aggregate_metadata(
    train_result: CollectionResult,
    val_result: CollectionResult,
    teachers: list[BaseTeacher],
) -> dict[str, Any]:
    all_rows = train_result.rows + val_result.rows
    all_assistant_payloads = [json.loads(row["messages"][2]["content"]) for row in all_rows]
    all_candidate_counts = [len(row.get("candidate_intersections", [])) for row in all_rows]
    all_target_counts = [len(payload.get("target_intersections", [])) for payload in all_assistant_payloads]
    return {
        "num_train_rows": len(train_result.rows),
        "num_val_rows": len(val_result.rows),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "teacher_sources": [teacher.metadata.to_dict() for teacher in teachers],
        "rows_per_city": counter_dict([row["city_id"] for row in all_rows]),
        "rows_per_scenario": counter_dict([row["scenario"] for row in all_rows]),
        "rows_per_district_type": counter_dict([row["district_type"] for row in all_rows]),
        "controller_family_counts": counter_dict([row["controller_family"] for row in all_rows]),
        "average_candidate_intersections_count": average(all_candidate_counts),
        "average_target_intersections_count": average(all_target_counts),
        "duplicate_rows_removed": train_result.duplicate_rows_removed + val_result.duplicate_rows_removed,
        "train_stats": build_split_diagnostics(train_result),
        "val_stats": build_split_diagnostics(val_result),
    }


def print_split_diagnostics(split_name: str, stats: dict[str, Any]) -> None:
    print(f"[{split_name}] rows={stats['rows']}")
    print(f"[{split_name}] rows_per_scenario={json.dumps(stats['rows_per_scenario'], sort_keys=True)}")
    print(f"[{split_name}] rows_per_district_type={json.dumps(stats['rows_per_district_type'], sort_keys=True)}")
    print(f"[{split_name}] rows_per_city={json.dumps(stats['rows_per_city'], sort_keys=True)}")
    print(f"[{split_name}] summary_length={json.dumps(stats['summary_length'], sort_keys=True)}")
    print(f"[{split_name}] assistant_json_length={json.dumps(stats['assistant_json_length'], sort_keys=True)}")
    print(f"[{split_name}] candidate_pool_size={json.dumps(stats['candidate_pool_size'], sort_keys=True)}")
    print(
        f"[{split_name}] target_intersections_count="
        f"{json.dumps(stats['target_intersections_count'], sort_keys=True)}"
    )
    print(f"[{split_name}] strategy_distribution={json.dumps(stats['strategy_distribution'], sort_keys=True)}")
    print(
        f"[{split_name}] priority_corridor_distribution="
        f"{json.dumps(stats['priority_corridor_distribution'], sort_keys=True)}"
    )
    print(f"[{split_name}] phase_bias_distribution={json.dumps(stats['phase_bias_distribution'], sort_keys=True)}")
    print(
        f"[{split_name}] duration_steps_distribution="
        f"{json.dumps(stats['duration_steps_distribution'], sort_keys=True)}"
    )
    print(f"[{split_name}] assistant_uniqueness_ratio={stats['assistant_uniqueness_ratio']:.3f}")
    print(
        f"[{split_name}] duplicate_rows_removed={stats['duplicate_rows_removed']} "
        f"low_signal_rows_removed={stats['low_signal_rows_removed']}"
    )


def main() -> None:
    args = parse_args()
    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    dataset.generate_default_splits()
    allowed_scenarios = parse_scenarios(args.scenarios)
    teachers = resolve_teachers(args)

    checkpoint_env_configs = [
        teacher.env_config for teacher in teachers if teacher.env_config is not None
    ]
    if checkpoint_env_configs[1:] and any(
        config != checkpoint_env_configs[0] for config in checkpoint_env_configs[1:]
    ):
        raise ValueError("Teacher checkpoint env configs differ. Use one DQN family per generation run.")
    env_config = checkpoint_env_configs[0] if checkpoint_env_configs else default_env_config()

    train_specs = build_balanced_scenario_specs(dataset, "train", args.cities, allowed_scenarios)
    val_specs = build_balanced_scenario_specs(dataset, "val", args.cities, allowed_scenarios)

    train_result = collect_split_rows("train", args.num_train, train_specs, teachers, env_config, args)
    val_result = collect_split_rows("val", args.num_val, val_specs, teachers, env_config, args)

    train_keys = {row_key(row) for row in train_result.rows}
    val_keys = {row_key(row) for row in val_result.rows}
    overlap = train_keys & val_keys
    if overlap:
        raise ValueError("Duplicate rows detected across train and val splits.")

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_result.rows)
    write_jsonl(output_dir / "val.jsonl", val_result.rows)

    metadata = aggregate_metadata(train_result, val_result, teachers)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print_split_diagnostics("train", metadata["train_stats"])
    print_split_diagnostics("val", metadata["val_stats"])
    print(f"[done] wrote {output_dir / 'train.jsonl'}")
    print(f"[done] wrote {output_dir / 'val.jsonl'}")
    print(f"[done] wrote {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
