from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from district_llm.generate_dataset import build_env, generate_examples_for_episode
from district_llm.schema import DistrictAction
from district_llm.teachers import BaseTeacher, build_teacher, parse_teacher_spec
from env.observation_builder import ObservationConfig
from env.reward import RewardConfig
from env.traffic_env import EnvConfig
from training.dataset import CityFlowDataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SYSTEM_PROMPT = (
    "You are a district traffic coordinator that outputs structured district guidance "
    "for RL traffic controllers. Return only valid JSON with fields: strategy, "
    "priority_corridor, target_intersections, phase_bias, duration_steps."
)
DEFAULT_OUTPUT_DIR = "data/district_llm_dataset_v1"
SCHEMA_VERSION = "district_action_v1_messages_v1"


@dataclass(frozen=True)
class SplitStats:
    action_distribution: dict[str, int]
    summary_length: dict[str, float]
    rows_per_scenario: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small pilot district-LLM chat dataset."
    )
    parser.add_argument("--num-train", type=int, default=400)
    parser.add_argument("--num-val", type=int, default=80)
    parser.add_argument("--cities", type=int, default=3)
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Either 'all' or a comma-separated scenario list such as normal,accident,event_spike.",
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--top-k-congested", type=int, default=3)
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
        teachers.append(
            build_teacher(
                controller_type=controller_type,
                checkpoint=checkpoint,
                fixed_green_time=args.fixed_green_time,
                seed=args.seed,
                device=args.device,
            )
        )
    return sorted(
        teachers,
        key=lambda teacher: (teacher.metadata.controller_family != "dqn", teacher.metadata.controller_type),
    )


def parse_scenarios(raw_value: str) -> set[str] | None:
    if raw_value.strip().lower() == "all":
        return None
    return {
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    }


def build_scenario_specs(
    dataset: CityFlowDataset,
    split_name: str,
    max_cities: int,
    allowed_scenarios: set[str] | None,
) -> list[Any]:
    city_ids = dataset.load_split(split_name)[:max_cities]
    scenario_specs = []
    for city_id in city_ids:
        for scenario_name in dataset.scenarios_for_city(city_id):
            if allowed_scenarios is not None and scenario_name not in allowed_scenarios:
                continue
            scenario_specs.append(dataset.build_scenario_spec(city_id, scenario_name))
    if not scenario_specs:
        raise ValueError(f"No scenario specs found for split={split_name}.")
    return scenario_specs


def make_message_row(example: dict[str, Any]) -> dict[str, Any]:
    assistant_content = json.dumps(example["response_json"], sort_keys=True)
    user_content = extract_summary_text(example["prompt"])

    row = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
    }
    validate_row(row)
    return row


def extract_summary_text(prompt: str) -> str:
    state_marker = "### DISTRICT STATE\n"
    decision_marker = "\n\n### DECISION"
    if state_marker not in prompt:
        return prompt.strip()
    summary_block = prompt.split(state_marker, 1)[1]
    summary_block = summary_block.split(decision_marker, 1)[0].strip()
    return f"### DISTRICT STATE\n{summary_block}"


def validate_row(row: dict[str, Any]) -> None:
    messages = row.get("messages", [])
    if len(messages) != 3:
        raise ValueError("Each row must contain exactly 3 chat messages.")
    if not messages[1]["content"].strip():
        raise ValueError("User summary content is empty.")
    payload = json.loads(messages[2]["content"])
    DistrictAction.from_dict(payload)


def row_key(row: dict[str, Any]) -> str:
    return json.dumps(row["messages"], sort_keys=True, separators=(",", ":"))


def summary_stats(rows: list[dict[str, Any]]) -> SplitStats:
    action_counter = Counter()
    scenario_counter = Counter()
    summary_lengths: list[int] = []

    for row in rows:
        assistant_payload = json.loads(row["messages"][2]["content"])
        action_counter[str(assistant_payload["strategy"])] += 1
        scenario_counter[str(row["scenario"])] += 1
        summary_lengths.append(len(row["messages"][1]["content"]))

    if not summary_lengths:
        length_stats = {"min": 0.0, "max": 0.0, "avg": 0.0}
    else:
        length_stats = {
            "min": float(min(summary_lengths)),
            "max": float(max(summary_lengths)),
            "avg": float(sum(summary_lengths) / len(summary_lengths)),
        }
    return SplitStats(
        action_distribution=dict(sorted(action_counter.items())),
        summary_length=length_stats,
        rows_per_scenario=dict(sorted(scenario_counter.items())),
    )


def collect_split_rows(
    split_name: str,
    target_rows: int,
    scenario_specs: list[Any],
    teachers: list[BaseTeacher],
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    pending_non_dqn: list[dict[str, Any]] = []
    episode_index = 0
    max_rounds = max(10, target_rows * 2)
    progress = (
        tqdm(
            total=target_rows,
            desc=f"{split_name} rows",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )

    try:
        while len(rows) < target_rows and episode_index < max_rounds:
            scenario_spec = scenario_specs[episode_index % len(scenario_specs)]
            if progress is not None:
                progress.set_postfix_str(
                    f"episode={episode_index} city={scenario_spec.city_id} scenario={scenario_spec.scenario_name}"
                )
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
                    episode_index=episode_index,
                )
                for example in examples:
                    row = make_message_row(example)
                    key = row_key(row)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    if example["controller_family"] == "dqn":
                        rows.append(row)
                        if progress is not None:
                            progress.update(1)
                    else:
                        pending_non_dqn.append(row)
                    if len(rows) >= target_rows:
                        break
                if len(rows) >= target_rows:
                    break
            episode_index += 1
    finally:
        if progress is not None:
            progress.close()

    if len(rows) < target_rows:
        for row in pending_non_dqn:
            if len(rows) >= target_rows:
                break
            rows.append(row)

    if len(rows) < target_rows:
        raise RuntimeError(
            f"Unable to collect {target_rows} unique rows for split={split_name}. "
            f"Collected {len(rows)} rows."
        )

    return rows[:target_rows]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def write_metadata(
    path: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    teachers: list[BaseTeacher],
) -> None:
    metadata = {
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "teacher_sources": [teacher.metadata.to_dict() for teacher in teachers],
        "schema_version": SCHEMA_VERSION,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_stats(split_name: str, rows: list[dict[str, Any]]) -> None:
    stats = summary_stats(rows)
    print(f"[{split_name}] rows={len(rows)}")
    print(f"[{split_name}] action_distribution={json.dumps(stats.action_distribution, sort_keys=True)}")
    print(f"[{split_name}] summary_length={json.dumps(stats.summary_length, sort_keys=True)}")
    print(f"[{split_name}] rows_per_scenario={json.dumps(stats.rows_per_scenario, sort_keys=True)}")


def validate_unique_rows(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    train_keys = [row_key(row) for row in train_rows]
    val_keys = [row_key(row) for row in val_rows]
    if len(train_keys) != len(set(train_keys)):
        raise ValueError("Duplicate rows detected inside train split.")
    if len(val_keys) != len(set(val_keys)):
        raise ValueError("Duplicate rows detected inside val split.")
    overlap = set(train_keys) & set(val_keys)
    if overlap:
        raise ValueError("Duplicate rows detected across train and val splits.")


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
        raise ValueError("Teacher checkpoint env configs differ. Use one DQN family per pilot run.")
    env_config = checkpoint_env_configs[0] if checkpoint_env_configs else default_env_config()

    train_specs = build_scenario_specs(
        dataset=dataset,
        split_name="train",
        max_cities=args.cities,
        allowed_scenarios=allowed_scenarios,
    )
    val_specs = build_scenario_specs(
        dataset=dataset,
        split_name="val",
        max_cities=args.cities,
        allowed_scenarios=allowed_scenarios,
    )

    train_rows = collect_split_rows(
        split_name="train",
        target_rows=args.num_train,
        scenario_specs=train_specs,
        teachers=teachers,
        env_config=env_config,
        args=args,
    )
    val_rows = collect_split_rows(
        split_name="val",
        target_rows=args.num_val,
        scenario_specs=val_specs,
        teachers=teachers,
        env_config=env_config,
        args=args,
    )
    validate_unique_rows(train_rows, val_rows)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_metadata(
        output_dir / "metadata.json",
        train_rows=train_rows,
        val_rows=val_rows,
        teachers=teachers,
    )

    print_stats("train", train_rows)
    print_stats("val", val_rows)
    print(f"[done] wrote {output_dir / 'train.jsonl'}")
    print(f"[done] wrote {output_dir / 'val.jsonl'}")
    print(f"[done] wrote {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
