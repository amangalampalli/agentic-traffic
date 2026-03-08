from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from district_llm.derivation import DistrictWindowData, LocalIntersectionAction, derive_district_action
from district_llm.prompting import format_district_prompt, format_sft_text
from district_llm.summary_builder import DistrictStateSummaryBuilder
from district_llm.teachers import BaseTeacher, build_teacher, parse_teacher_spec
from env.observation_builder import ObservationConfig
from env.reward import RewardConfig
from env.traffic_env import EnvConfig, TrafficEnv
from training.cityflow_dataset import CityFlowDataset, ScenarioSpec


@dataclass
class _WindowBuffer:
    start_summary: Any
    controller_actions: list[LocalIntersectionAction] = field(default_factory=list)
    step_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate district-LLM SFT data from CityFlow rollouts."
    )
    parser.add_argument(
        "--controller",
        default="queue_greedy",
        choices=("rl_checkpoint", "hold", "fixed", "random", "queue_greedy"),
        help="Single controller source used when --teacher-spec is not provided.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--teacher-spec",
        action="append",
        default=[],
        help="Repeatable source spec, e.g. rl_checkpoint=artifacts/dqn_shared/best_validation.pt or fixed.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--decision-interval",
        "--district-decision-interval",
        dest="district_decision_interval",
        type=int,
        default=10,
        help="District-LLM decision interval in local-controller decision steps.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--city-id", default=None)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fixed-green-time", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--top-k-congested", type=int, default=3)
    parser.add_argument("--max-candidate-intersections", type=int, default=6)
    parser.add_argument("--max-target-intersections", type=int, default=3)
    parser.add_argument("--use-checkpoint-env-config", action="store_true")

    parser.add_argument("--env-decision-interval", type=int, default=5)
    parser.add_argument("--simulator-interval", type=int, default=1)
    parser.add_argument("--min-green-time", type=int, default=10)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--max-episode-seconds", type=int, default=None)
    parser.add_argument("--max-incoming-lanes", type=int, default=16)
    parser.add_argument("--count-scale", type=float, default=20.0)
    parser.add_argument("--elapsed-time-scale", type=float, default=60.0)
    parser.add_argument("--disable-district-context", action="store_true")
    parser.add_argument("--disable-outgoing-congestion", action="store_true")
    parser.add_argument("--reward-variant", default="wait_queue_throughput")
    parser.add_argument("--waiting-weight", type=float, default=1.0)
    parser.add_argument("--vehicle-weight", type=float, default=0.1)
    parser.add_argument("--pressure-weight", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--disable-lane-reward-normalization", action="store_true")
    parser.add_argument("--reward-clip", type=float, default=5.0)
    parser.add_argument("--queue-delta-weight", type=float, default=2.0)
    parser.add_argument("--wait-delta-weight", type=float, default=4.0)
    parser.add_argument("--queue-level-weight", type=float, default=0.5)
    parser.add_argument("--wait-level-weight", type=float, default=1.0)
    parser.add_argument("--throughput-weight", type=float, default=0.1)
    parser.add_argument("--imbalance-weight", type=float, default=0.1)
    parser.add_argument("--reward-delta-clip", type=float, default=2.0)
    parser.add_argument("--reward-level-normalizer", type=float, default=10.0)
    parser.add_argument("--throughput-normalizer", type=float, default=2.0)
    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> EnvConfig:
    return EnvConfig(
        simulator_interval=args.simulator_interval,
        decision_interval=args.env_decision_interval,
        min_green_time=args.min_green_time,
        thread_num=args.thread_num,
        max_episode_seconds=args.max_episode_seconds,
        observation=ObservationConfig(
            max_incoming_lanes=args.max_incoming_lanes,
            count_scale=args.count_scale,
            elapsed_time_scale=args.elapsed_time_scale,
            include_outgoing_congestion=not args.disable_outgoing_congestion,
            include_district_context=not args.disable_district_context,
            include_district_type_feature=True,
        ),
        reward=RewardConfig(
            variant=args.reward_variant,
            waiting_weight=args.waiting_weight,
            vehicle_weight=args.vehicle_weight,
            pressure_weight=args.pressure_weight,
            reward_scale=args.reward_scale,
            normalize_by_lane_count=not args.disable_lane_reward_normalization,
            clip_reward=args.reward_clip,
            queue_delta_weight=args.queue_delta_weight,
            wait_delta_weight=args.wait_delta_weight,
            queue_level_weight=args.queue_level_weight,
            wait_level_weight=args.wait_level_weight,
            throughput_weight=args.throughput_weight,
            imbalance_weight=args.imbalance_weight,
            delta_clip=args.reward_delta_clip,
            level_normalizer=args.reward_level_normalizer,
            throughput_normalizer=args.throughput_normalizer,
        ),
    )


def build_env(env_config: EnvConfig, scenario_spec: ScenarioSpec) -> TrafficEnv:
    return TrafficEnv(
        city_id=scenario_spec.city_id,
        scenario_name=scenario_spec.scenario_name,
        city_dir=scenario_spec.city_dir,
        scenario_dir=scenario_spec.scenario_dir,
        config_path=scenario_spec.config_path,
        roadnet_path=scenario_spec.roadnet_path,
        district_map_path=scenario_spec.district_map_path,
        metadata_path=scenario_spec.metadata_path,
        env_config=env_config,
    )


def resolve_teachers(args: argparse.Namespace) -> list[BaseTeacher]:
    teacher_specs = list(args.teacher_spec)
    if not teacher_specs:
        teacher_specs = [args.controller if args.controller != "rl_checkpoint" else f"rl_checkpoint={args.checkpoint}"]

    teachers = []
    for spec in teacher_specs:
        controller_type, checkpoint = parse_teacher_spec(spec)
        if controller_type == "rl_checkpoint":
            checkpoint = checkpoint or args.checkpoint
        teachers.append(
            build_teacher(
                controller_type=controller_type,
                checkpoint=checkpoint,
                fixed_green_time=args.fixed_green_time,
                seed=args.seed,
                device=args.device,
            )
        )
    return teachers


def resolve_env_config(args: argparse.Namespace, teachers: list[BaseTeacher]) -> EnvConfig:
    env_config = build_env_config(args)
    if not args.use_checkpoint_env_config:
        return env_config

    checkpoint_env_configs = [
        teacher.env_config for teacher in teachers if teacher.env_config is not None
    ]
    if not checkpoint_env_configs:
        return env_config

    first_payload = checkpoint_env_configs[0]
    assert first_payload is not None
    for item in checkpoint_env_configs[1:]:
        if item != first_payload:
            raise ValueError("Checkpoint teachers use different env configs. Generate separate datasets.")
    return first_payload


def sample_scenario(
    dataset: CityFlowDataset,
    rng: random.Random,
    split: str,
    city_id: str | None,
    scenario_name: str | None,
) -> ScenarioSpec:
    if city_id and scenario_name:
        return dataset.build_scenario_spec(city_id, scenario_name)
    return dataset.sample_scenario(
        split_name=split,
        rng=rng,
        city_id=city_id,
        scenario_name=scenario_name,
    )


def extract_step_actions(
    env: TrafficEnv,
    observation_batch: dict[str, Any],
    next_observation_batch: dict[str, Any],
    actions: np.ndarray,
) -> dict[str, list[LocalIntersectionAction]]:
    grouped: dict[str, list[LocalIntersectionAction]] = {district_id: [] for district_id in env.districts}
    lane_vehicle_count = env.adapter.get_lane_vehicle_count()

    for index, intersection_id in enumerate(observation_batch["intersection_ids"]):
        district_id = observation_batch["district_ids"][index]
        grouped[district_id].append(
            LocalIntersectionAction(
                intersection_id=intersection_id,
                district_id=district_id,
                action=int(actions[index]),
                current_phase=int(observation_batch["current_phase"][index]),
                next_phase=int(next_observation_batch["current_phase"][index]),
                queue_total=float(np.asarray(observation_batch["incoming_counts"][index], dtype=np.float32).sum()),
                wait_total=float(np.asarray(observation_batch["incoming_waiting"][index], dtype=np.float32).sum()),
                outgoing_load=float(
                    sum(
                        float(lane_vehicle_count.get(lane_id, 0))
                        for lane_id in env.intersections[intersection_id].outgoing_lanes
                    )
                ),
                is_boundary=bool(env.intersections[intersection_id].is_boundary),
            )
        )
    return grouped


def generate_examples_for_episode(
    env: TrafficEnv,
    teacher: BaseTeacher,
    district_interval: int,
    top_k_congested: int,
    max_candidate_intersections: int,
    max_target_intersections: int,
    episode_index: int,
) -> list[dict[str, Any]]:
    summary_builder = DistrictStateSummaryBuilder(
        top_k=top_k_congested,
        candidate_limit=max_candidate_intersections,
    )
    observation_batch = env.reset()
    summary_builder.reset()
    current_summaries = summary_builder.build_all(env, observation_batch)
    windows = {
        district_id: _WindowBuffer(start_summary=summary)
        for district_id, summary in current_summaries.items()
    }
    samples: list[dict[str, Any]] = []
    done = False
    window_index = 0

    while not done:
        actions = teacher.act(observation_batch)
        next_observation_batch, rewards, done, info = env.step(actions)
        del rewards, info
        step_actions = extract_step_actions(env, observation_batch, next_observation_batch, actions)
        next_summaries = summary_builder.build_all(env, next_observation_batch)

        for district_id, buffer in windows.items():
            buffer.controller_actions.extend(step_actions[district_id])
            buffer.step_count += 1
            should_emit = buffer.step_count >= district_interval or done
            if not should_emit:
                continue

            end_summary = next_summaries[district_id]
            window_data = DistrictWindowData(
                district_id=district_id,
                start_summary=buffer.start_summary,
                end_summary=end_summary,
                controller_actions=list(buffer.controller_actions),
                step_count=buffer.step_count,
            )
            action = derive_district_action(
                window_data=window_data,
                max_target_intersections=max_target_intersections,
            )
            prompt = format_district_prompt(
                buffer.start_summary,
                max_target_intersections=max_target_intersections,
                allow_only_visible_candidates=True,
            )
            samples.append(
                {
                    "text": format_sft_text(
                        buffer.start_summary,
                        action,
                        max_target_intersections=max_target_intersections,
                        allow_only_visible_candidates=True,
                    ),
                    "prompt": prompt,
                    "response_json": action.to_dict(),
                    "state": buffer.start_summary.to_dict(),
                    "candidate_intersections": buffer.start_summary.to_dict().get("candidate_intersections", []),
                    "window_summary": window_data.to_dict(),
                    "city_id": env.city_id,
                    "district_id": district_id,
                    "district_type": env.districts[district_id].district_type,
                    "scenario": env.scenario_name,
                    "controller_type": teacher.metadata.controller_type,
                    "controller_id": teacher.metadata.controller_id,
                    "controller_family": teacher.metadata.controller_family,
                    "teacher_algorithm": teacher.metadata.teacher_algorithm,
                    "checkpoint_path": teacher.metadata.checkpoint_path,
                    "episode_index": int(episode_index),
                    "window_index": int(window_index),
                    "decision_interval": int(district_interval),
                    "sim_time": int(buffer.start_summary.sim_time),
                }
            )
            windows[district_id] = _WindowBuffer(start_summary=end_summary)
            window_index += 1

        observation_batch = next_observation_batch

    return samples


def append_jsonl(path: Path, records: list[dict[str, Any]], append: bool) -> None:
    mode = "a" if append else "w"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    dataset.generate_default_splits()
    rng = random.Random(args.seed)
    teachers = resolve_teachers(args)
    env_config = resolve_env_config(args, teachers)

    output_path = Path(args.output)
    write_mode_append = bool(args.append)

    for episode_index in range(args.episodes):
        scenario_spec = sample_scenario(
            dataset=dataset,
            rng=rng,
            split=args.split,
            city_id=args.city_id,
            scenario_name=args.scenario_name,
        )
        episode_records: list[dict[str, Any]] = []
        for teacher in teachers:
            env = build_env(env_config=env_config, scenario_spec=scenario_spec)
            episode_records.extend(
                generate_examples_for_episode(
                    env=env,
                    teacher=teacher,
                    district_interval=args.district_decision_interval,
                    top_k_congested=args.top_k_congested,
                    max_candidate_intersections=args.max_candidate_intersections,
                    max_target_intersections=args.max_target_intersections,
                    episode_index=episode_index,
                )
            )
        append_jsonl(output_path, episode_records, append=write_mode_append)
        write_mode_append = True
        print(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "city_id": scenario_spec.city_id,
                    "scenario_name": scenario_spec.scenario_name,
                    "records_written": len(episode_records),
                }
            )
        )


if __name__ == "__main__":
    main()
