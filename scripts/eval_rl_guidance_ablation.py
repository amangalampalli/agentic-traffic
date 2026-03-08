from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
import sys

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from district_llm.heuristic_guidance import HeuristicGuidanceConfig
from district_llm.inference import DistrictLLMInference
from district_llm.repair import RepairConfig
from district_llm.rl_guidance_wrapper import (
    BIAS_DECAY_SCHEDULES,
    DistrictGuidedRLController,
    FixedRLPolicyAdapter,
    GATING_MODES,
    GuidanceInfluenceConfig,
    HeuristicGuidanceProvider,
    LLMGuidanceProvider,
    WRAPPER_MODES,
    guidance_config_payload,
)
from district_llm.summary_builder import DistrictStateSummaryBuilder
from env.observation_builder import ObservationConfig
from env.reward import RewardConfig
from env.traffic_env import EnvConfig
from env.utils import load_json
from training.cityflow_dataset import CityFlowDataset, ScenarioSpec
from training.train_local_policy import build_env


MODE_CHOICES: tuple[str, ...] = (
    "rl_only",
    "rl_heuristic",
    "rl_llm",
)


@dataclass(frozen=True)
class EpisodePlan:
    city_id: str
    scenario: str
    seed: int
    episode_id: int
    simulator_seed: int
    scenario_spec: ScenarioSpec
    seeded_scenario_spec: ScenarioSpec

    def pairing_key(self) -> tuple[str, str, int, int]:
        return (self.city_id, self.scenario, self.seed, self.episode_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "city_id": self.city_id,
            "scenario": self.scenario,
            "seed": int(self.seed),
            "episode_id": int(self.episode_id),
            "simulator_seed": int(self.simulator_seed),
            "config_path": str(self.seeded_scenario_spec.config_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fixed DQN checkpoint under rl_only, rl_heuristic, and "
            "rl_llm district-guidance modes without changing the RL weights."
        )
    )
    parser.add_argument(
        "--rl-checkpoint",
        required=True,
        help="Path to the fixed DQN checkpoint used for all modes.",
    )
    parser.add_argument(
        "--llm-model-path",
        default=None,
        help="Model or adapter path used when rl_llm modes are enabled.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODE_CHOICES,
        default=["rl_only", "rl_heuristic", "rl_llm"],
    )
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 13])
    parser.add_argument(
        "--max-episode-seconds",
        type=int,
        default=None,
        help="Optional override for scenario horizon. Useful for cheap smoke tests.",
    )
    parser.add_argument("--guidance-refresh-steps", type=int, default=10)
    parser.add_argument("--guidance-persistence-steps", type=int, default=3)
    parser.add_argument("--bias-strength", type=float, default=0.12)
    parser.add_argument(
        "--targeted-bias-strength",
        "--target-only-bias-strength",
        dest="targeted_bias_strength",
        type=float,
        default=0.18,
    )
    parser.add_argument("--corridor-bias-strength", type=float, default=0.05)
    parser.add_argument("--max-guidance-duration", type=int, default=10)
    parser.add_argument("--max-intersections-affected", type=int, default=3)
    parser.add_argument(
        "--gating-mode",
        choices=GATING_MODES,
        default="always_on",
    )
    parser.add_argument("--min-avg-queue-for-guidance", type=float, default=150.0)
    parser.add_argument("--min-queue-imbalance-for-guidance", type=float, default=20.0)
    parser.add_argument(
        "--require-incident-or-spillback",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow-guidance-in-normal-conditions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-bias-decay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bias-decay-schedule",
        choices=BIAS_DECAY_SCHEDULES,
        default="linear",
    )
    parser.add_argument(
        "--apply-global-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--apply-target-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--wrapper-modes",
        "--wrapper-mode",
        dest="wrapper_modes",
        nargs="+",
        choices=WRAPPER_MODES,
        default=["target_only_soft"],
    )
    parser.add_argument(
        "--allow-only-visible-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-target-intersections", type=int, default=3)
    parser.add_argument(
        "--fallback-on-empty-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fallback-mode",
        choices=("heuristic", "hold", "none"),
        default="heuristic",
    )
    parser.add_argument(
        "--fallback-policy",
        choices=("no_op", "hold_previous", "heuristic_weak"),
        default="hold_previous",
    )
    parser.add_argument(
        "--log-guidance-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="artifacts/rl_guidance_eval")
    parser.add_argument(
        "--save-step-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-guidance-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "rl_llm" in args.modes and not args.llm_model_path:
        raise ValueError("--llm-model-path is required when rl_llm is selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeded_config_root = output_dir / "seeded_configs"
    seeded_config_root.mkdir(parents=True, exist_ok=True)

    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    dataset.generate_default_splits()
    scenario_specs = resolve_scenario_specs(dataset=dataset, args=args)
    episode_plans = build_episode_plans(
        scenario_specs=scenario_specs,
        seeds=args.seeds,
        num_episodes=args.num_episodes,
        seeded_config_root=seeded_config_root,
    )

    rl_policy = FixedRLPolicyAdapter(
        checkpoint_path=args.rl_checkpoint,
        device=args.device,
    )
    env_config = rl_policy.env_config or default_env_config()
    if args.max_episode_seconds is not None:
        env_config = EnvConfig(
            simulator_interval=env_config.simulator_interval,
            decision_interval=env_config.decision_interval,
            min_green_time=env_config.min_green_time,
            thread_num=env_config.thread_num,
            max_episode_seconds=int(args.max_episode_seconds),
            observation=env_config.observation,
            reward=env_config.reward,
        )
    controllers = build_mode_controllers(
        args=args,
        rl_policy=rl_policy,
    )
    controller_specs = list(controllers.items())

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    guidance_trace_rows: list[dict[str, Any]] = []
    total_runs = len(episode_plans) * len(controller_specs)
    progress = tqdm(total=total_runs, desc="RL guidance eval", unit="run")
    try:
        for plan_index, plan in enumerate(episode_plans, start=1):
            tqdm.write(
                "[episode-plan] "
                f"{plan_index}/{len(episode_plans)} "
                f"city={plan.city_id} "
                f"scenario={plan.scenario} "
                f"seed={plan.seed} "
                f"episode_id={plan.episode_id} "
                f"simulator_seed={plan.simulator_seed}"
            )
            for mode_label, controller in controller_specs:
                progress.set_postfix_str(
                    f"mode={mode_label} city={plan.city_id} scenario={plan.scenario} seed={plan.seed}"
                )
                episode_row, mode_step_rows, mode_trace_rows = run_episode(
                    plan=plan,
                    mode_label=mode_label,
                    controller=controller,
                    env_config=env_config,
                    save_step_metrics=args.save_step_metrics,
                    save_guidance_traces=args.save_guidance_traces,
                )
                episode_rows.append(episode_row)
                step_rows.extend(mode_step_rows)
                guidance_trace_rows.extend(mode_trace_rows)
                tqdm.write(
                    "[episode-result] "
                    f"mode={mode_label} "
                    f"return={episode_row['total_return']:.3f} "
                    f"avg_queue={episode_row['avg_queue']:.3f} "
                    f"avg_wait={episode_row['avg_wait']:.3f} "
                    f"throughput={episode_row['throughput']:.3f}"
                )
                progress.update(1)
    finally:
        progress.close()

    config_payload = build_config_payload(
        args=args,
        env_config=env_config,
        episode_plans=episode_plans,
    )
    summary_payload = build_summary_payload(
        episode_rows=episode_rows,
        config_payload=config_payload,
    )

    write_json(output_dir / "config.json", config_payload)
    write_json(output_dir / "summary.json", summary_payload)
    write_csv_rows(output_dir / "episode_metrics.csv", episode_rows)
    write_jsonl(output_dir / "episode_metrics.jsonl", episode_rows)
    episode_parquet_written = try_write_parquet(output_dir / "episode_metrics.parquet", episode_rows)

    if args.save_step_metrics:
        write_csv_rows(output_dir / "step_metrics.csv", step_rows)
        write_jsonl(output_dir / "step_metrics.jsonl", step_rows)
        try_write_parquet(output_dir / "step_metrics.parquet", step_rows)

    if args.save_guidance_traces:
        write_jsonl(output_dir / "guidance_traces.jsonl", guidance_trace_rows)

    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    if not episode_parquet_written:
        print(
            "[warning] episode_metrics.parquet was not written because neither pyarrow nor pandas "
            "is available in the current Python environment."
        )


def resolve_scenario_specs(
    dataset: CityFlowDataset,
    args: argparse.Namespace,
) -> list[ScenarioSpec]:
    city_ids = list(args.cities) if args.cities else dataset.load_split(args.split)
    scenario_specs: list[ScenarioSpec] = []
    for city_id in city_ids:
        available_scenarios = dataset.scenarios_for_city(city_id)
        if not available_scenarios:
            raise ValueError(f"No scenarios found for city '{city_id}'.")
        requested_scenarios = list(args.scenarios) if args.scenarios else available_scenarios
        for scenario_name in requested_scenarios:
            if scenario_name not in available_scenarios:
                raise ValueError(
                    f"Scenario '{scenario_name}' is not available for city '{city_id}'. "
                    f"Available scenarios: {available_scenarios}"
                )
            scenario_specs.append(dataset.build_scenario_spec(city_id, scenario_name))
    if not scenario_specs:
        raise ValueError("No scenario specs were resolved for evaluation.")
    return scenario_specs


def build_episode_plans(
    scenario_specs: list[ScenarioSpec],
    seeds: list[int],
    num_episodes: int,
    seeded_config_root: Path,
) -> list[EpisodePlan]:
    plans: list[EpisodePlan] = []
    for scenario_spec in scenario_specs:
        for seed in seeds:
            for episode_id in range(num_episodes):
                simulator_seed = int(seed) * 1000 + int(episode_id)
                seeded_spec = build_seeded_scenario_spec(
                    scenario_spec=scenario_spec,
                    simulator_seed=simulator_seed,
                    seeded_config_root=seeded_config_root,
                )
                plans.append(
                    EpisodePlan(
                        city_id=scenario_spec.city_id,
                        scenario=scenario_spec.scenario_name,
                        seed=int(seed),
                        episode_id=int(episode_id),
                        simulator_seed=int(simulator_seed),
                        scenario_spec=scenario_spec,
                        seeded_scenario_spec=seeded_spec,
                    )
                )
    return plans


def build_seeded_scenario_spec(
    scenario_spec: ScenarioSpec,
    simulator_seed: int,
    seeded_config_root: Path,
) -> ScenarioSpec:
    payload = load_json(scenario_spec.config_path)
    payload["seed"] = int(simulator_seed)
    destination_dir = (
        seeded_config_root
        / scenario_spec.city_id
        / scenario_spec.scenario_name
        / f"seed_{int(simulator_seed):08d}"
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    config_path = destination_dir / "config.json"
    write_json(config_path, payload)
    return ScenarioSpec(
        city_id=scenario_spec.city_id,
        scenario_name=scenario_spec.scenario_name,
        city_dir=scenario_spec.city_dir,
        scenario_dir=scenario_spec.scenario_dir,
        config_path=config_path,
        roadnet_path=scenario_spec.roadnet_path,
        district_map_path=scenario_spec.district_map_path,
        metadata_path=scenario_spec.metadata_path,
    )


def build_mode_controllers(
    args: argparse.Namespace,
    rl_policy: FixedRLPolicyAdapter,
) -> dict[str, DistrictGuidedRLController]:
    heuristic_provider = HeuristicGuidanceProvider(
        config=HeuristicGuidanceConfig(
            max_target_intersections=args.max_target_intersections,
        )
    )

    llm_inference = None
    if "rl_llm" in args.modes:
        llm_inference = DistrictLLMInference(
            model_name_or_path=args.llm_model_path,
            device=args.device,
            repair_config=RepairConfig(
                allow_only_visible_candidates=args.allow_only_visible_candidates,
                max_target_intersections=args.max_target_intersections,
                fallback_on_empty_targets=args.fallback_on_empty_targets,
                fallback_mode=args.fallback_mode,
            ),
        )

    controllers: dict[str, DistrictGuidedRLController] = {}
    for mode in args.modes:
        if mode == "rl_only":
            controllers["rl_only"] = DistrictGuidedRLController(
                policy=rl_policy,
                mode_source="rl_only",
                summary_builder=None,
                guidance_provider=None,
                influence_config=GuidanceInfluenceConfig(
                    wrapper_mode="no_op",
                    bias_strength=args.bias_strength,
                    target_only_bias_strength=args.targeted_bias_strength,
                    corridor_bias_strength=args.corridor_bias_strength,
                    max_intersections_affected=args.max_intersections_affected,
                    guidance_refresh_steps=args.guidance_refresh_steps,
                    guidance_persistence_steps=args.guidance_persistence_steps,
                    max_guidance_duration=args.max_guidance_duration,
                    apply_global_bias=False,
                    apply_target_only=True,
                    gating_mode=args.gating_mode,
                    min_avg_queue_for_guidance=args.min_avg_queue_for_guidance,
                    min_queue_imbalance_for_guidance=args.min_queue_imbalance_for_guidance,
                    require_incident_or_spillback=args.require_incident_or_spillback,
                    allow_guidance_in_normal_conditions=args.allow_guidance_in_normal_conditions,
                    enable_bias_decay=args.enable_bias_decay,
                    bias_decay_schedule=args.bias_decay_schedule,
                    fallback_policy=args.fallback_policy,
                    log_guidance_debug=False,
                ),
                heuristic_provider=None,
            )
            continue

        for wrapper_mode in args.wrapper_modes:
            influence_config = GuidanceInfluenceConfig(
                wrapper_mode=wrapper_mode,
                bias_strength=args.bias_strength,
                target_only_bias_strength=args.targeted_bias_strength,
                corridor_bias_strength=args.corridor_bias_strength,
                max_intersections_affected=args.max_intersections_affected,
                guidance_refresh_steps=args.guidance_refresh_steps,
                guidance_persistence_steps=args.guidance_persistence_steps,
                max_guidance_duration=args.max_guidance_duration,
                apply_global_bias=args.apply_global_bias,
                apply_target_only=args.apply_target_only,
                gating_mode=args.gating_mode,
                min_avg_queue_for_guidance=args.min_avg_queue_for_guidance,
                min_queue_imbalance_for_guidance=args.min_queue_imbalance_for_guidance,
                require_incident_or_spillback=args.require_incident_or_spillback,
                allow_guidance_in_normal_conditions=args.allow_guidance_in_normal_conditions,
                enable_bias_decay=args.enable_bias_decay,
                bias_decay_schedule=args.bias_decay_schedule,
                fallback_policy=args.fallback_policy,
                log_guidance_debug=args.log_guidance_debug,
            )
            summary_builder = DistrictStateSummaryBuilder(
                top_k=3,
                candidate_limit=max(6, int(args.max_target_intersections)),
            )
            label = f"{mode}+{wrapper_mode}"
            if mode == "rl_heuristic":
                controllers[label] = DistrictGuidedRLController(
                    policy=rl_policy,
                    mode_source=mode,
                    summary_builder=summary_builder,
                    guidance_provider=heuristic_provider,
                    influence_config=influence_config,
                    heuristic_provider=heuristic_provider,
                )
                continue

            assert llm_inference is not None
            controllers[label] = DistrictGuidedRLController(
                policy=rl_policy,
                mode_source=mode,
                summary_builder=summary_builder,
                guidance_provider=LLMGuidanceProvider(
                    inference=llm_inference,
                    max_new_tokens=args.max_new_tokens,
                ),
                influence_config=influence_config,
                heuristic_provider=heuristic_provider,
            )
    return controllers


def run_episode(
    plan: EpisodePlan,
    mode_label: str,
    controller: DistrictGuidedRLController,
    env_config: EnvConfig,
    save_step_metrics: bool,
    save_guidance_traces: bool,
    show_step_progress: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = build_env(env_config, plan.seeded_scenario_spec)
    controller.reset()
    observation_batch = env.reset()
    estimated_steps = max(
        1,
        int(np.ceil(float(env.max_episode_seconds) / float(env.env_config.decision_interval))),
    )
    step_progress = None
    if show_step_progress:
        step_progress = tqdm(
            total=estimated_steps,
            desc=f"{mode_label} {plan.city_id}/{plan.scenario} seed={plan.seed}",
            unit="step",
            leave=False,
        )
    episode_started = perf_counter()
    wrapper_runtime_seconds = 0.0
    guidance_runtime_seconds = 0.0
    guidance_refresh_count = 0
    fallback_used_count = 0
    invalid_guidance_count = 0
    repaired_guidance_count = 0
    action_changes_vs_base = 0
    decision_steps = 0
    queue_series: list[float] = []
    wait_series: list[float] = []
    running_vehicle_series: list[float] = []
    spillback_total = 0.0
    spillback_event_steps = 0
    step_rows: list[dict[str, Any]] = []
    guidance_trace_rows: list[dict[str, Any]] = []
    scenario_metadata = load_scenario_metadata(plan.scenario_spec)
    done = False

    try:
        while not done:
            action_batch = controller.act(env=env, observation_batch=observation_batch)
            wrapper_runtime_seconds += float(action_batch.runtime_seconds)
            decision_steps += 1
            action_changes_vs_base += int(np.sum(action_batch.actions != action_batch.base_actions))

            for trace in action_batch.refresh_traces:
                guidance_refresh_count += 1
                guidance_runtime_seconds += float(trace.guidance.get("runtime_seconds", 0.0))
                fallback_used_count += int(trace.fallback_used)
                invalid_guidance_count += int(trace.guidance.get("invalid_before_repair", False))
                repaired_guidance_count += int(trace.guidance.get("repair_applied", False))
                if save_guidance_traces:
                    guidance_trace_rows.append(
                        build_guidance_trace_row(
                            plan=plan,
                            mode_label=mode_label,
                            trace=trace,
                            controller=controller,
                        )
                    )

            next_observation_batch, rewards, done, info = env.step(action_batch.actions)
            metrics = info["metrics"]
            queue_total = safe_float(metrics.get("total_incoming_vehicles"))
            wait_total = safe_float(metrics.get("total_waiting_vehicles"))
            queue_series.append(queue_total)
            wait_series.append(wait_total)
            running_vehicle_series.append(safe_float(metrics.get("running_vehicles")))

            spillback_intersections = estimate_spillback_intersections(observation_batch)
            spillback_total += float(spillback_intersections)
            spillback_event_steps += int(spillback_intersections > 0)

            if step_progress is not None:
                step_progress.set_postfix_str(
                    " ".join(
                        [
                            f"sim={int(info['sim_time'])}",
                            f"queue={queue_total:.0f}",
                            f"wait={wait_total:.0f}",
                            f"thr={safe_float(metrics.get('throughput')):.0f}",
                            f"refresh={guidance_refresh_count}",
                            f"fallback={fallback_used_count}",
                        ]
                    )
                )
                step_progress.update(1)

            if save_step_metrics:
                step_rows.append(
                    build_step_row(
                        plan=plan,
                        mode_label=mode_label,
                        info=info,
                        action_batch=action_batch,
                        controller=controller,
                        spillback_intersections=spillback_intersections,
                        rewards=rewards,
                    )
                )

            observation_batch = next_observation_batch
    finally:
        if step_progress is not None:
            step_progress.close()

    episode_runtime_seconds = perf_counter() - episode_started
    final_metrics = env.last_info["metrics"]
    wrapper_debug = controller.episode_debug_summary()
    episode_row = {
        "mode": mode_label,
        "mode_source": controller.mode_source,
        "wrapper_mode": controller.influence_config.wrapper_mode,
        "city_id": plan.city_id,
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        "episode_id": int(plan.episode_id),
        "simulator_seed": int(plan.simulator_seed),
        "total_return": safe_float(env.total_episode_return),
        "mean_return": safe_float(env.episode_return),
        "avg_queue": average(queue_series),
        "max_queue": max_or_zero(queue_series),
        "total_queue": float(sum(queue_series)),
        "avg_wait": average(wait_series),
        "max_wait": max_or_zero(wait_series),
        "total_wait": float(sum(wait_series)),
        "throughput": safe_float(final_metrics.get("throughput")),
        "travel_time": safe_float(final_metrics.get("average_travel_time")),
        "avg_running_vehicles": average(running_vehicle_series),
        "max_running_vehicles": max_or_zero(running_vehicle_series),
        "spillback_count": float(spillback_total),
        "spillback_event_steps": float(spillback_event_steps),
        "incident_scenario": float(bool(scenario_metadata.get("blocked_roads"))),
        "construction_scenario": float(scenario_metadata.get("name") == "construction"),
        "event_scenario": float(bool(scenario_metadata.get("event_district"))),
        "overload_scenario": float(bool(scenario_metadata.get("overload_district"))),
        "num_guidance_refreshes": float(guidance_refresh_count),
        "runtime_seconds": float(episode_runtime_seconds),
        "guidance_inference_seconds": float(guidance_runtime_seconds),
        "wrapper_runtime_seconds": float(wrapper_runtime_seconds),
        "fallback_used_count": float(fallback_used_count),
        "invalid_guidance_count": float(invalid_guidance_count),
        "repaired_guidance_count": float(repaired_guidance_count),
        "action_changes_vs_base": float(action_changes_vs_base),
        "decision_steps": float(env.decision_step_count),
        "num_controlled_intersections": safe_float(final_metrics.get("num_controlled_intersections")),
    }
    episode_row.update(wrapper_debug)
    return episode_row, step_rows, guidance_trace_rows


def build_step_row(
    plan: EpisodePlan,
    mode_label: str,
    info: dict[str, Any],
    action_batch,
    controller: DistrictGuidedRLController,
    spillback_intersections: int,
    rewards: np.ndarray,
) -> dict[str, Any]:
    metrics = info["metrics"]
    active_guidance = controller.active_guidance_snapshot()
    return {
        "mode": mode_label,
        "mode_source": controller.mode_source,
        "wrapper_mode": controller.influence_config.wrapper_mode,
        "city_id": plan.city_id,
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        "episode_id": int(plan.episode_id),
        "simulator_seed": int(plan.simulator_seed),
        "step": int(info["decision_step"]),
        "sim_time": int(info["sim_time"]),
        "queue": safe_float(metrics.get("total_incoming_vehicles")),
        "wait": safe_float(metrics.get("total_waiting_vehicles")),
        "throughput": safe_float(metrics.get("throughput")),
        "travel_time": safe_float(metrics.get("average_travel_time")),
        "running_vehicles": safe_float(metrics.get("running_vehicles")),
        "step_total_reward": float(np.asarray(rewards, dtype=np.float32).sum()),
        "action_changes_vs_base": int(np.sum(action_batch.actions != action_batch.base_actions)),
        "mean_abs_q_bias": float(np.abs(action_batch.q_bias).mean()),
        "spillback_intersections": int(spillback_intersections),
        "active_guidance_count": int(len(active_guidance)),
        "active_guidance_json": json.dumps(active_guidance, sort_keys=True),
        "selected_target_intersections_json": json.dumps(
            collect_target_intersections(active_guidance),
            sort_keys=True,
        ),
        "phase_bias_json": json.dumps(
            {
                district_id: payload.get("phase_bias")
                for district_id, payload in sorted(active_guidance.items())
            },
            sort_keys=True,
        ),
        "priority_corridor_json": json.dumps(
            {
                district_id: payload.get("priority_corridor")
                for district_id, payload in sorted(active_guidance.items())
            },
            sort_keys=True,
        ),
    }


def build_guidance_trace_row(
    plan: EpisodePlan,
    mode_label: str,
    trace,
    controller: DistrictGuidedRLController,
) -> dict[str, Any]:
    payload = trace.to_dict()
    payload.update(
        {
            "mode": mode_label,
            "wrapper_mode": controller.influence_config.wrapper_mode,
            "city_id": plan.city_id,
            "scenario": plan.scenario,
            "seed": int(plan.seed),
            "episode_id": int(plan.episode_id),
            "simulator_seed": int(plan.simulator_seed),
            "influence_config": guidance_config_payload(controller.influence_config),
        }
    )
    return payload


def estimate_spillback_intersections(observation_batch: dict[str, Any]) -> int:
    incoming_totals = np.asarray(observation_batch["incoming_counts"], dtype=np.float32).sum(axis=1)
    outgoing_load = np.asarray(observation_batch["outgoing_congestion"], dtype=np.float32)
    boundary_mask = np.asarray(observation_batch["boundary_mask"], dtype=np.float32) > 0.0
    spillback_mask = outgoing_load >= np.maximum(8.0, incoming_totals * 0.5)
    boundary_spillback = boundary_mask & (outgoing_load >= np.maximum(4.0, incoming_totals * 0.4))
    return int(np.sum(spillback_mask | boundary_spillback))


def collect_target_intersections(active_guidance: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    return {
        district_id: list(payload.get("target_intersections", []))
        for district_id, payload in sorted(active_guidance.items())
    }


def load_scenario_metadata(scenario_spec: ScenarioSpec) -> dict[str, Any]:
    metadata_path = scenario_spec.scenario_dir / "scenario_metadata.json"
    return load_json(metadata_path) if metadata_path.exists() else {}


def build_summary_payload(
    episode_rows: list[dict[str, Any]],
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    key_metrics = (
        "total_return",
        "mean_return",
        "avg_queue",
        "avg_wait",
        "throughput",
        "travel_time",
        "spillback_count",
        "fallback_used_count",
        "invalid_guidance_count",
        "repaired_guidance_count",
        "num_steps_guidance_blocked_by_gate",
        "num_guidance_refreshes_blocked_by_gate",
        "mean_bias_magnitude",
        "max_bias_magnitude",
        "avg_num_targeted_intersections",
        "avg_num_affected_intersections",
        "percent_steps_with_active_guidance",
        "num_noop_guidance_events",
    )
    metrics_by_mode: dict[str, Any] = {}
    for mode in sorted({str(row["mode"]) for row in episode_rows}):
        mode_rows = [row for row in episode_rows if row["mode"] == mode]
        metrics_by_mode[mode] = {
            metric_name: distribution_summary(
                [safe_float(row.get(metric_name)) for row in mode_rows]
            )
            for metric_name in key_metrics
        }
        metrics_by_mode[mode]["num_episodes"] = int(len(mode_rows))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": config_payload["comparison_scope"],
        "pairing_keys": ["city_id", "scenario", "seed", "episode_id"],
        "metrics_by_mode": metrics_by_mode,
        "analysis_summary": build_analysis_summary(episode_rows),
    }


def build_config_payload(
    args: argparse.Namespace,
    env_config: EnvConfig,
    episode_plans: list[EpisodePlan],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rl_checkpoint": str(args.rl_checkpoint),
        "llm_model_path": args.llm_model_path,
        "modes": list(args.modes),
        "wrapper_modes": list(args.wrapper_modes),
        "comparison_scope": {
            "num_episode_plans": int(len(episode_plans)),
            "cities": sorted({plan.city_id for plan in episode_plans}),
            "scenarios": sorted({plan.scenario for plan in episode_plans}),
            "seeds": sorted({int(plan.seed) for plan in episode_plans}),
            "episodes_per_seed": int(args.num_episodes),
            "total_mode_runs": int(
                len(
                    [
                        1
                        for mode in args.modes
                        for _wrapper in ([None] if mode == "rl_only" else args.wrapper_modes)
                    ]
                )
                * len(episode_plans)
            ),
        },
        "episode_plans": [plan.to_dict() for plan in episode_plans],
        "guidance_influence_config": guidance_config_payload(
            GuidanceInfluenceConfig(
                wrapper_mode=args.wrapper_modes[0],
                bias_strength=args.bias_strength,
                target_only_bias_strength=args.targeted_bias_strength,
                corridor_bias_strength=args.corridor_bias_strength,
                max_intersections_affected=args.max_intersections_affected,
                guidance_refresh_steps=args.guidance_refresh_steps,
                guidance_persistence_steps=args.guidance_persistence_steps,
                max_guidance_duration=args.max_guidance_duration,
                apply_global_bias=args.apply_global_bias,
                apply_target_only=args.apply_target_only,
                gating_mode=args.gating_mode,
                min_avg_queue_for_guidance=args.min_avg_queue_for_guidance,
                min_queue_imbalance_for_guidance=args.min_queue_imbalance_for_guidance,
                require_incident_or_spillback=args.require_incident_or_spillback,
                allow_guidance_in_normal_conditions=args.allow_guidance_in_normal_conditions,
                enable_bias_decay=args.enable_bias_decay,
                bias_decay_schedule=args.bias_decay_schedule,
                fallback_policy=args.fallback_policy,
                log_guidance_debug=args.log_guidance_debug,
            )
        ),
        "repair_config": asdict(
            RepairConfig(
                allow_only_visible_candidates=args.allow_only_visible_candidates,
                max_target_intersections=args.max_target_intersections,
                fallback_on_empty_targets=args.fallback_on_empty_targets,
                fallback_mode=args.fallback_mode,
            )
        ),
        "heuristic_config": asdict(
            HeuristicGuidanceConfig(
                max_target_intersections=args.max_target_intersections,
            )
        ),
        "env_config": env_config_to_payload(env_config),
        "save_step_metrics": bool(args.save_step_metrics),
        "save_guidance_traces": bool(args.save_guidance_traces),
    }


def build_analysis_summary(episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not episode_rows:
        return {}
    rl_only_rows = [row for row in episode_rows if row["mode_source"] == "rl_only"]
    rl_only_return = average([safe_float(row.get("total_return")) for row in rl_only_rows])
    guided_modes = sorted({str(row["mode"]) for row in episode_rows if row["mode_source"] != "rl_only"})
    ranked_guided = []
    mode_source_rows: dict[str, list[dict[str, Any]]] = {}
    for row in episode_rows:
        mode_source_rows.setdefault(str(row["mode_source"]), []).append(row)
    for row_mode in guided_modes:
        mode_rows = [item for item in episode_rows if item["mode"] == row_mode]
        return_summary = distribution_summary([safe_float(item.get("total_return")) for item in mode_rows])
        queue_summary = distribution_summary([safe_float(item.get("avg_queue")) for item in mode_rows])
        fallback_summary = distribution_summary([safe_float(item.get("fallback_used_count")) for item in mode_rows])
        affected_summary = distribution_summary(
            [safe_float(item.get("avg_num_affected_intersections")) for item in mode_rows]
        )
        steps_summary = distribution_summary(
            [safe_float(item.get("percent_steps_with_active_guidance")) for item in mode_rows]
        )
        gate_block_summary = distribution_summary(
            [safe_float(item.get("num_steps_guidance_blocked_by_gate")) for item in mode_rows]
        )
        ranked_guided.append(
            {
                "mode": row_mode,
                "mode_source": str(mode_rows[0]["mode_source"]),
                "wrapper_mode": str(mode_rows[0]["wrapper_mode"]),
                "mean_total_return": return_summary["mean"],
                "return_delta_vs_rl_only": return_summary["mean"] - rl_only_return,
                "mean_avg_queue": queue_summary["mean"],
                "mean_fallback_used_count": fallback_summary["mean"],
                "mean_avg_num_affected_intersections": affected_summary["mean"],
                "mean_percent_steps_with_active_guidance": steps_summary["mean"],
                "mean_num_steps_guidance_blocked_by_gate": gate_block_summary["mean"],
            }
        )
    ranked_guided.sort(key=lambda item: item["mean_total_return"], reverse=True)

    mode_source_summary = {
        mode_source: {
            "mean_total_return": distribution_summary(
                [safe_float(item.get("total_return")) for item in rows]
            )["mean"],
            "mean_fallback_used_count": distribution_summary(
                [safe_float(item.get("fallback_used_count")) for item in rows]
            )["mean"],
            "mean_avg_num_affected_intersections": distribution_summary(
                [safe_float(item.get("avg_num_affected_intersections")) for item in rows]
            )["mean"],
            "mean_num_steps_guidance_blocked_by_gate": distribution_summary(
                [safe_float(item.get("num_steps_guidance_blocked_by_gate")) for item in rows]
            )["mean"],
        }
        for mode_source, rows in sorted(mode_source_rows.items())
    }

    heuristic_vs_llm_by_wrapper: list[dict[str, Any]] = []
    shared_wrappers = sorted(
        {
            str(row["wrapper_mode"])
            for row in episode_rows
            if row["mode_source"] in {"rl_heuristic", "rl_llm"}
        }
    )
    for wrapper_mode in shared_wrappers:
        heuristic_rows = [
            row for row in episode_rows if row["mode_source"] == "rl_heuristic" and row["wrapper_mode"] == wrapper_mode
        ]
        llm_rows = [
            row for row in episode_rows if row["mode_source"] == "rl_llm" and row["wrapper_mode"] == wrapper_mode
        ]
        if not heuristic_rows or not llm_rows:
            continue
        heuristic_return = distribution_summary([safe_float(row.get("total_return")) for row in heuristic_rows])["mean"]
        llm_return = distribution_summary([safe_float(row.get("total_return")) for row in llm_rows])["mean"]
        heuristic_vs_llm_by_wrapper.append(
            {
                "wrapper_mode": wrapper_mode,
                "heuristic_mean_total_return": heuristic_return,
                "llm_mean_total_return": llm_return,
                "llm_minus_heuristic_return": llm_return - heuristic_return,
            }
        )

    aggressive_modes = [
        item
        for item in ranked_guided
        if item["wrapper_mode"] in {"global_soft", "current_legacy"}
    ]
    conservative_modes = [
        item
        for item in ranked_guided
        if item["wrapper_mode"] in {"no_op", "target_only_soft", "target_only_medium", "corridor_soft"}
    ]
    return {
        "rl_only_mean_total_return": rl_only_return,
        "guided_mode_rankings": ranked_guided,
        "least_degrading_guided_mode": ranked_guided[0] if ranked_guided else None,
        "mode_source_summary": mode_source_summary,
        "heuristic_vs_llm_by_wrapper": heuristic_vs_llm_by_wrapper,
        "conservative_guidance_modes": conservative_modes,
        "aggressive_guidance_modes": aggressive_modes,
    }


def distribution_summary(values: list[float]) -> dict[str, float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    array = np.asarray(filtered, dtype=np.float64)
    return {
        "count": float(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(median(filtered)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


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


def env_config_to_payload(env_config: EnvConfig) -> dict[str, Any]:
    return {
        "simulator_interval": env_config.simulator_interval,
        "decision_interval": env_config.decision_interval,
        "min_green_time": env_config.min_green_time,
        "thread_num": env_config.thread_num,
        "max_episode_seconds": env_config.max_episode_seconds,
        "observation": asdict(env_config.observation),
        "reward": asdict(env_config.reward),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format_csv_value(row.get(key))
                    for key in fieldnames
                }
            )


def try_write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    json_ready_rows = [to_jsonable(row) for row in rows]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(json_ready_rows)
        pq.write_table(table, path)
        return True
    except Exception:
        pass

    try:
        import pandas as pd

        frame = pd.DataFrame(json_ready_rows)
        frame.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def format_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return to_jsonable(value)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def max_or_zero(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
