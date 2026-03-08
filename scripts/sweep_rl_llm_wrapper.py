from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from district_llm.heuristic_guidance import HeuristicGuidanceConfig
from district_llm.inference import DistrictLLMInference
from district_llm.repair import RepairConfig
from district_llm.rl_guidance_wrapper import (
    BIAS_DECAY_SCHEDULES,
    GATING_MODES,
    DistrictGuidedRLController,
    FixedRLPolicyAdapter,
    GuidanceInfluenceConfig,
    HeuristicGuidanceProvider,
    LLMGuidanceProvider,
    guidance_config_payload,
)
from district_llm.summary_builder import DistrictStateSummaryBuilder
from env.traffic_env import EnvConfig
from scripts.eval_rl_guidance_ablation import (
    build_episode_plans,
    default_env_config,
    distribution_summary,
    env_config_to_payload,
    resolve_scenario_specs,
    run_episode,
    safe_float,
    try_write_parquet,
    write_csv_rows,
    write_json,
    write_jsonl,
)
from training.cityflow_dataset import CityFlowDataset


PRESET_CHOICES: tuple[str, ...] = (
    "strength_only",
    "strength_and_targets",
    "strength_targets_gating",
    "full_conservative",
)
DEFAULT_CITIES: tuple[str, ...] = ("city_0001",)
DEFAULT_SCENARIOS: tuple[str, ...] = ("normal",)


@dataclass(frozen=True)
class SweepConfigSpec:
    config_id: str
    description: str
    wrapper_mode: str
    bias_strength: float
    target_only_bias_strength: float
    corridor_bias_strength: float
    max_intersections_affected: int
    guidance_persistence_steps: int
    guidance_refresh_steps: int
    max_guidance_duration: int
    gating_mode: str
    min_avg_queue_for_guidance: float
    min_queue_imbalance_for_guidance: float
    require_incident_or_spillback: bool
    allow_guidance_in_normal_conditions: bool
    enable_bias_decay: bool
    bias_decay_schedule: str
    fallback_policy: str
    is_reference: bool = False

    def to_influence_config(self) -> GuidanceInfluenceConfig:
        return GuidanceInfluenceConfig(
            wrapper_mode=self.wrapper_mode,
            bias_strength=self.bias_strength,
            target_only_bias_strength=self.target_only_bias_strength,
            corridor_bias_strength=self.corridor_bias_strength,
            max_intersections_affected=self.max_intersections_affected,
            guidance_refresh_steps=self.guidance_refresh_steps,
            guidance_persistence_steps=self.guidance_persistence_steps,
            max_guidance_duration=self.max_guidance_duration,
            apply_global_bias=False,
            apply_target_only=True,
            gating_mode=self.gating_mode,
            min_avg_queue_for_guidance=self.min_avg_queue_for_guidance,
            min_queue_imbalance_for_guidance=self.min_queue_imbalance_for_guidance,
            require_incident_or_spillback=self.require_incident_or_spillback,
            allow_guidance_in_normal_conditions=self.allow_guidance_in_normal_conditions,
            enable_bias_decay=self.enable_bias_decay,
            bias_decay_schedule=self.bias_decay_schedule,
            fallback_policy=self.fallback_policy,
            log_guidance_debug=False,
        ).validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cheap paired hyperparameter sweep for the fixed RL + district LLM wrapper. "
            "The RL checkpoint and LLM checkpoint stay fixed; only inference-time wrapper "
            "hyperparameters are varied."
        )
    )
    parser.add_argument("--rl-checkpoint", required=True)
    parser.add_argument("--llm-model-path", required=True)
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 13])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument(
        "--max-episode-seconds",
        type=int,
        default=300,
        help="Cheap default horizon for wrapper tuning sweeps.",
    )
    parser.add_argument(
        "--preset",
        choices=PRESET_CHOICES,
        default="strength_targets_gating",
    )
    parser.add_argument("--guidance-refresh-steps", type=int, default=10)
    parser.add_argument("--max-guidance-duration", type=int, default=10)
    parser.add_argument("--queue-threshold", type=float, default=150.0)
    parser.add_argument("--imbalance-threshold", type=float, default=20.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="artifacts/rl_llm_wrapper_sweep")
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
        default="no_op",
    )
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
    parser.add_argument(
        "--bias-decay-schedule",
        choices=BIAS_DECAY_SCHEDULES,
        default="linear",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
        num_episodes=args.episodes_per_seed,
        seeded_config_root=seeded_config_root,
    )
    sweep_configs = build_sweep_configs(args)

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

    rl_only_controller = build_rl_only_controller(
        rl_policy=rl_policy,
        guidance_refresh_steps=args.guidance_refresh_steps,
        max_guidance_duration=args.max_guidance_duration,
    )
    guided_controllers = build_guided_controllers(
        args=args,
        rl_policy=rl_policy,
        sweep_configs=sweep_configs,
    )

    sweep_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    rl_only_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    guidance_trace_rows: list[dict[str, Any]] = []

    total_runs = len(episode_plans) * (1 + len(sweep_configs))
    progress = tqdm(total=total_runs, desc="RL+LLM wrapper sweep", unit="run")
    try:
        for plan_index, plan in enumerate(episode_plans, start=1):
            progress.set_postfix_str(
                f"rl_only city={plan.city_id} scenario={plan.scenario} seed={plan.seed}"
            )
            rl_only_row, rl_only_step_rows, rl_only_trace_rows = run_episode(
                plan=plan,
                mode_label="rl_only",
                controller=rl_only_controller,
                env_config=env_config,
                save_step_metrics=args.save_step_metrics,
                save_guidance_traces=args.save_guidance_traces,
                show_step_progress=False,
            )
            rl_only_row = augment_rl_only_row(rl_only_row)
            rl_only_rows.append(rl_only_row)
            if args.save_step_metrics:
                step_rows.extend(
                    augment_auxiliary_rows(
                        rows=rl_only_step_rows,
                        config_id="rl_only",
                        config_spec=None,
                    )
                )
            if args.save_guidance_traces:
                guidance_trace_rows.extend(
                    augment_auxiliary_rows(
                        rows=rl_only_trace_rows,
                        config_id="rl_only",
                        config_spec=None,
                    )
                )
            progress.update(1)

            for config in sweep_configs:
                controller = guided_controllers[config.config_id]
                progress.set_postfix_str(
                    f"{config.config_id} city={plan.city_id} scenario={plan.scenario} seed={plan.seed}"
                )
                episode_row, mode_step_rows, mode_trace_rows = run_episode(
                    plan=plan,
                    mode_label=config.config_id,
                    controller=controller,
                    env_config=env_config,
                    save_step_metrics=args.save_step_metrics,
                    save_guidance_traces=args.save_guidance_traces,
                    show_step_progress=False,
                )
                episode_row = augment_guided_row(episode_row, config)
                sweep_rows.append(episode_row)
                paired_rows.append(build_paired_row(guided_row=episode_row, rl_only_row=rl_only_row))
                if args.save_step_metrics:
                    step_rows.extend(
                        augment_auxiliary_rows(
                            rows=mode_step_rows,
                            config_id=config.config_id,
                            config_spec=config,
                        )
                    )
                if args.save_guidance_traces:
                    guidance_trace_rows.extend(
                        augment_auxiliary_rows(
                            rows=mode_trace_rows,
                            config_id=config.config_id,
                            config_spec=config,
                        )
                    )
                progress.update(1)
            tqdm.write(
                "[sweep-plan] "
                f"{plan_index}/{len(episode_plans)} "
                f"city={plan.city_id} scenario={plan.scenario} seed={plan.seed} complete"
            )
    finally:
        progress.close()

    ranking_rows = build_config_rankings(
        paired_rows=paired_rows,
        sweep_configs=sweep_configs,
    )
    summary_report = build_summary_report(
        paired_rows=paired_rows,
        ranking_rows=ranking_rows,
        rl_only_rows=rl_only_rows,
        args=args,
        sweep_configs=sweep_configs,
    )
    config_payload = build_config_payload(
        args=args,
        env_config=env_config,
        episode_plans=episode_plans,
        sweep_configs=sweep_configs,
    )

    write_json(output_dir / "config.json", config_payload)
    write_csv_rows(output_dir / "sweep_results.csv", sweep_rows)
    write_jsonl(output_dir / "sweep_results.jsonl", sweep_rows)
    try_write_parquet(output_dir / "sweep_results.parquet", sweep_rows)
    write_csv_rows(output_dir / "paired_episode_metrics.csv", paired_rows)
    write_jsonl(output_dir / "paired_episode_metrics.jsonl", paired_rows)
    try_write_parquet(output_dir / "paired_episode_metrics.parquet", paired_rows)
    write_csv_rows(output_dir / "rl_only_episode_metrics.csv", rl_only_rows)
    write_json(output_dir / "ranking.json", ranking_rows)
    write_json(output_dir / "summary_report.json", summary_report)

    if args.save_step_metrics:
        write_csv_rows(output_dir / "step_metrics.csv", step_rows)
        write_jsonl(output_dir / "step_metrics.jsonl", step_rows)
        try_write_parquet(output_dir / "step_metrics.parquet", step_rows)
    if args.save_guidance_traces:
        write_jsonl(output_dir / "guidance_traces.jsonl", guidance_trace_rows)

    print(json.dumps(summary_report, indent=2, sort_keys=True))


def build_rl_only_controller(
    rl_policy: FixedRLPolicyAdapter,
    guidance_refresh_steps: int,
    max_guidance_duration: int,
) -> DistrictGuidedRLController:
    return DistrictGuidedRLController(
        policy=rl_policy,
        mode_source="rl_only",
        summary_builder=None,
        guidance_provider=None,
        influence_config=GuidanceInfluenceConfig(
            wrapper_mode="no_op",
            bias_strength=0.0,
            target_only_bias_strength=0.0,
            corridor_bias_strength=0.0,
            max_intersections_affected=1,
            guidance_refresh_steps=guidance_refresh_steps,
            guidance_persistence_steps=1,
            max_guidance_duration=max_guidance_duration,
            gating_mode="always_on",
            enable_bias_decay=False,
            fallback_policy="no_op",
        ),
        heuristic_provider=None,
    )


def build_guided_controllers(
    args: argparse.Namespace,
    rl_policy: FixedRLPolicyAdapter,
    sweep_configs: list[SweepConfigSpec],
) -> dict[str, DistrictGuidedRLController]:
    repair_config = RepairConfig(
        allow_only_visible_candidates=args.allow_only_visible_candidates,
        max_target_intersections=args.max_target_intersections,
        fallback_on_empty_targets=args.fallback_on_empty_targets,
        fallback_mode=args.fallback_mode,
    )
    llm_inference = DistrictLLMInference(
        model_name_or_path=args.llm_model_path,
        device=args.device,
        repair_config=repair_config,
    )
    heuristic_provider = HeuristicGuidanceProvider(
        config=HeuristicGuidanceConfig(
            max_target_intersections=args.max_target_intersections,
        )
    )
    llm_provider = LLMGuidanceProvider(
        inference=llm_inference,
        max_new_tokens=args.max_new_tokens,
    )
    controllers: dict[str, DistrictGuidedRLController] = {}
    for config in sweep_configs:
        controllers[config.config_id] = DistrictGuidedRLController(
            policy=rl_policy,
            mode_source="rl_llm",
            summary_builder=DistrictStateSummaryBuilder(
                top_k=3,
                candidate_limit=max(6, int(args.max_target_intersections)),
            ),
            guidance_provider=llm_provider,
            influence_config=config.to_influence_config(),
            heuristic_provider=heuristic_provider,
        )
    return controllers


def build_sweep_configs(args: argparse.Namespace) -> list[SweepConfigSpec]:
    configs: list[SweepConfigSpec] = [
        build_baseline_reference_config(args),
    ]
    if args.preset == "strength_only":
        for bias_strength in (0.025, 0.05, 0.075, 0.10):
            configs.append(
                build_target_only_soft_config(
                    args=args,
                    bias_strength=bias_strength,
                    max_intersections_affected=2,
                    guidance_persistence_steps=5,
                    gating_mode="queue_or_imbalance",
                    enable_bias_decay=False,
                )
            )
    elif args.preset == "strength_and_targets":
        for bias_strength in (0.025, 0.05, 0.075, 0.10):
            for max_intersections_affected in (1, 2):
                configs.append(
                    build_target_only_soft_config(
                        args=args,
                        bias_strength=bias_strength,
                        max_intersections_affected=max_intersections_affected,
                        guidance_persistence_steps=5,
                        gating_mode="queue_or_imbalance",
                        enable_bias_decay=False,
                    )
                )
    elif args.preset == "strength_targets_gating":
        for bias_strength in (0.025, 0.05, 0.075):
            for max_intersections_affected in (1, 2):
                for gating_mode in ("always_on", "incident_or_spillback", "queue_or_imbalance"):
                    configs.append(
                        build_target_only_soft_config(
                            args=args,
                            bias_strength=bias_strength,
                            max_intersections_affected=max_intersections_affected,
                            guidance_persistence_steps=5,
                            gating_mode=gating_mode,
                            enable_bias_decay=False,
                        )
                    )
    else:
        for bias_strength in (0.025, 0.05, 0.075):
            for max_intersections_affected in (1, 2):
                for gating_mode, guidance_persistence_steps, enable_bias_decay in (
                    ("queue_or_imbalance", 5, False),
                    ("queue_or_imbalance", 10, True),
                    ("incident_or_spillback", 5, False),
                    ("incident_or_spillback", 10, True),
                ):
                    configs.append(
                        build_target_only_soft_config(
                            args=args,
                            bias_strength=bias_strength,
                            max_intersections_affected=max_intersections_affected,
                            guidance_persistence_steps=guidance_persistence_steps,
                            gating_mode=gating_mode,
                            enable_bias_decay=enable_bias_decay,
                        )
                    )
    return dedupe_sweep_configs(configs)


def build_baseline_reference_config(args: argparse.Namespace) -> SweepConfigSpec:
    return SweepConfigSpec(
        config_id="baseline_current_soft",
        description="Current rl_llm + target_only_soft reference config from the smoke runs.",
        wrapper_mode="target_only_soft",
        bias_strength=0.12,
        target_only_bias_strength=0.18,
        corridor_bias_strength=0.05,
        max_intersections_affected=3,
        guidance_persistence_steps=3,
        guidance_refresh_steps=args.guidance_refresh_steps,
        max_guidance_duration=max(args.max_guidance_duration, 3),
        gating_mode="always_on",
        min_avg_queue_for_guidance=args.queue_threshold,
        min_queue_imbalance_for_guidance=args.imbalance_threshold,
        require_incident_or_spillback=False,
        allow_guidance_in_normal_conditions=True,
        enable_bias_decay=True,
        bias_decay_schedule=args.bias_decay_schedule,
        fallback_policy=args.fallback_policy,
        is_reference=True,
    )


def build_target_only_soft_config(
    args: argparse.Namespace,
    bias_strength: float,
    max_intersections_affected: int,
    guidance_persistence_steps: int,
    gating_mode: str,
    enable_bias_decay: bool,
) -> SweepConfigSpec:
    target_only_bias_strength = bias_strength
    corridor_bias_strength = 0.5 * bias_strength
    config_id = (
        f"bs{format_float_token(bias_strength)}"
        f"_aff{int(max_intersections_affected)}"
        f"_gate{gating_mode_token(gating_mode)}"
        f"_p{int(guidance_persistence_steps)}"
        f"_decay{int(enable_bias_decay)}"
    )
    return SweepConfigSpec(
        config_id=config_id,
        description=(
            "Curated conservative target_only_soft sweep config with locally tied target/corridor "
            "bias strengths."
        ),
        wrapper_mode="target_only_soft",
        bias_strength=float(bias_strength),
        target_only_bias_strength=float(target_only_bias_strength),
        corridor_bias_strength=float(corridor_bias_strength),
        max_intersections_affected=int(max_intersections_affected),
        guidance_persistence_steps=int(guidance_persistence_steps),
        guidance_refresh_steps=int(args.guidance_refresh_steps),
        max_guidance_duration=max(int(args.max_guidance_duration), int(guidance_persistence_steps)),
        gating_mode=gating_mode,
        min_avg_queue_for_guidance=float(args.queue_threshold),
        min_queue_imbalance_for_guidance=float(args.imbalance_threshold),
        require_incident_or_spillback=False,
        allow_guidance_in_normal_conditions=(gating_mode == "always_on"),
        enable_bias_decay=bool(enable_bias_decay),
        bias_decay_schedule=args.bias_decay_schedule,
        fallback_policy=args.fallback_policy,
        is_reference=False,
    )


def dedupe_sweep_configs(configs: list[SweepConfigSpec]) -> list[SweepConfigSpec]:
    deduped: list[SweepConfigSpec] = []
    seen_ids: set[str] = set()
    for config in configs:
        if config.config_id in seen_ids:
            continue
        deduped.append(config)
        seen_ids.add(config.config_id)
    return deduped


def augment_rl_only_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.update(
        {
            "config_id": "rl_only",
            "description": "Fixed RL policy with no district guidance.",
            "is_reference": True,
            "bias_strength": 0.0,
            "target_only_bias_strength": 0.0,
            "corridor_bias_strength": 0.0,
            "max_intersections_affected": 0,
            "guidance_persistence_steps": 0,
            "guidance_refresh_steps": 0,
            "max_guidance_duration": 0,
            "gating_mode": "always_on",
            "min_avg_queue_for_guidance": 0.0,
            "min_queue_imbalance_for_guidance": 0.0,
            "require_incident_or_spillback": False,
            "allow_guidance_in_normal_conditions": True,
            "enable_bias_decay": False,
            "bias_decay_schedule": "linear",
        }
    )
    return payload


def augment_guided_row(row: dict[str, Any], config: SweepConfigSpec) -> dict[str, Any]:
    payload = dict(row)
    payload.update(
        {
            "config_id": config.config_id,
            "description": config.description,
            "is_reference": bool(config.is_reference),
            "bias_strength": float(config.bias_strength),
            "target_only_bias_strength": float(config.target_only_bias_strength),
            "corridor_bias_strength": float(config.corridor_bias_strength),
            "max_intersections_affected": int(config.max_intersections_affected),
            "guidance_persistence_steps": int(config.guidance_persistence_steps),
            "guidance_refresh_steps": int(config.guidance_refresh_steps),
            "max_guidance_duration": int(config.max_guidance_duration),
            "gating_mode": config.gating_mode,
            "min_avg_queue_for_guidance": float(config.min_avg_queue_for_guidance),
            "min_queue_imbalance_for_guidance": float(config.min_queue_imbalance_for_guidance),
            "require_incident_or_spillback": bool(config.require_incident_or_spillback),
            "allow_guidance_in_normal_conditions": bool(config.allow_guidance_in_normal_conditions),
            "enable_bias_decay": bool(config.enable_bias_decay),
            "bias_decay_schedule": config.bias_decay_schedule,
        }
    )
    return payload


def augment_auxiliary_rows(
    rows: list[dict[str, Any]],
    config_id: str,
    config_spec: SweepConfigSpec | None,
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["config_id"] = config_id
        payload["is_reference"] = bool(config_spec.is_reference) if config_spec is not None else False
        if config_spec is not None:
            payload["gating_mode"] = config_spec.gating_mode
            payload["bias_strength"] = float(config_spec.bias_strength)
            payload["max_intersections_affected"] = int(config_spec.max_intersections_affected)
            payload["guidance_persistence_steps"] = int(config_spec.guidance_persistence_steps)
            payload["enable_bias_decay"] = bool(config_spec.enable_bias_decay)
        augmented.append(payload)
    return augmented


def build_paired_row(guided_row: dict[str, Any], rl_only_row: dict[str, Any]) -> dict[str, Any]:
    paired_row = dict(guided_row)
    paired_row.update(
        {
            "rl_only_total_return": safe_float(rl_only_row.get("total_return")),
            "rl_only_avg_queue": safe_float(rl_only_row.get("avg_queue")),
            "rl_only_avg_wait": safe_float(rl_only_row.get("avg_wait")),
            "rl_only_throughput": safe_float(rl_only_row.get("throughput")),
            "rl_only_travel_time": safe_float(rl_only_row.get("travel_time")),
            "total_return_delta_vs_rl_only": safe_float(guided_row.get("total_return"))
            - safe_float(rl_only_row.get("total_return")),
            "avg_queue_delta_vs_rl_only": safe_float(guided_row.get("avg_queue"))
            - safe_float(rl_only_row.get("avg_queue")),
            "avg_wait_delta_vs_rl_only": safe_float(guided_row.get("avg_wait"))
            - safe_float(rl_only_row.get("avg_wait")),
            "throughput_delta_vs_rl_only": safe_float(guided_row.get("throughput"))
            - safe_float(rl_only_row.get("throughput")),
            "travel_time_delta_vs_rl_only": safe_float(guided_row.get("travel_time"))
            - safe_float(rl_only_row.get("travel_time")),
        }
    )
    return paired_row


def build_config_rankings(
    paired_rows: list[dict[str, Any]],
    sweep_configs: list[SweepConfigSpec],
) -> list[dict[str, Any]]:
    rows_by_config = {
        config.config_id: [row for row in paired_rows if row["config_id"] == config.config_id]
        for config in sweep_configs
    }
    rankings: list[dict[str, Any]] = []
    config_lookup = {config.config_id: config for config in sweep_configs}
    for config_id, rows in rows_by_config.items():
        if not rows:
            continue
        config = config_lookup[config_id]
        summary = {
            "config_id": config_id,
            "description": config.description,
            "is_reference": bool(config.is_reference),
            "wrapper_mode": config.wrapper_mode,
            "bias_strength": float(config.bias_strength),
            "target_only_bias_strength": float(config.target_only_bias_strength),
            "corridor_bias_strength": float(config.corridor_bias_strength),
            "max_intersections_affected": int(config.max_intersections_affected),
            "guidance_persistence_steps": int(config.guidance_persistence_steps),
            "guidance_refresh_steps": int(config.guidance_refresh_steps),
            "gating_mode": config.gating_mode,
            "min_avg_queue_for_guidance": float(config.min_avg_queue_for_guidance),
            "min_queue_imbalance_for_guidance": float(config.min_queue_imbalance_for_guidance),
            "require_incident_or_spillback": bool(config.require_incident_or_spillback),
            "allow_guidance_in_normal_conditions": bool(config.allow_guidance_in_normal_conditions),
            "enable_bias_decay": bool(config.enable_bias_decay),
            "mean_total_return": distribution_summary(
                [safe_float(row.get("total_return")) for row in rows]
            )["mean"],
            "mean_return_delta_vs_rl_only": distribution_summary(
                [safe_float(row.get("total_return_delta_vs_rl_only")) for row in rows]
            )["mean"],
            "mean_avg_queue_delta_vs_rl_only": distribution_summary(
                [safe_float(row.get("avg_queue_delta_vs_rl_only")) for row in rows]
            )["mean"],
            "mean_avg_wait_delta_vs_rl_only": distribution_summary(
                [safe_float(row.get("avg_wait_delta_vs_rl_only")) for row in rows]
            )["mean"],
            "mean_throughput_delta_vs_rl_only": distribution_summary(
                [safe_float(row.get("throughput_delta_vs_rl_only")) for row in rows]
            )["mean"],
            "mean_travel_time_delta_vs_rl_only": distribution_summary(
                [safe_float(row.get("travel_time_delta_vs_rl_only")) for row in rows]
            )["mean"],
            "mean_percent_steps_with_active_guidance": distribution_summary(
                [safe_float(row.get("percent_steps_with_active_guidance")) for row in rows]
            )["mean"],
            "mean_avg_num_affected_intersections": distribution_summary(
                [safe_float(row.get("avg_num_affected_intersections")) for row in rows]
            )["mean"],
            "mean_avg_num_targeted_intersections": distribution_summary(
                [safe_float(row.get("avg_num_targeted_intersections")) for row in rows]
            )["mean"],
            "mean_num_steps_guidance_blocked_by_gate": distribution_summary(
                [safe_float(row.get("num_steps_guidance_blocked_by_gate")) for row in rows]
            )["mean"],
            "mean_fallback_used_count": distribution_summary(
                [safe_float(row.get("fallback_used_count")) for row in rows]
            )["mean"],
            "mean_invalid_guidance_count": distribution_summary(
                [safe_float(row.get("invalid_guidance_count")) for row in rows]
            )["mean"],
            "num_episodes": int(len(rows)),
        }
        summary["beats_rl_only"] = bool(summary["mean_return_delta_vs_rl_only"] >= 0.0)
        rankings.append(summary)
    rankings.sort(
        key=lambda item: (
            float(item["mean_return_delta_vs_rl_only"]),
            float(item["mean_throughput_delta_vs_rl_only"]),
            -float(item["mean_avg_queue_delta_vs_rl_only"]),
            -float(item["mean_avg_wait_delta_vs_rl_only"]),
        ),
        reverse=True,
    )
    return rankings


def build_summary_report(
    paired_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    rl_only_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    sweep_configs: list[SweepConfigSpec],
) -> dict[str, Any]:
    rl_only_mean_total_return = distribution_summary(
        [safe_float(row.get("total_return")) for row in rl_only_rows]
    )["mean"]
    top_5 = ranking_rows[:5]
    best_config = ranking_rows[0] if ranking_rows else None
    configs_beating_rl_only = [row for row in ranking_rows if row["beats_rl_only"]]

    bias_effects = group_rankings_by_parameter(ranking_rows, "bias_strength")
    affected_intersections_effects = group_rankings_by_parameter(ranking_rows, "max_intersections_affected")
    gating_effects = group_rankings_by_parameter(ranking_rows, "gating_mode")
    persistence_effects = group_rankings_by_parameter(ranking_rows, "guidance_persistence_steps")
    decay_effects = group_rankings_by_parameter(ranking_rows, "enable_bias_decay")

    best_bias = best_group_value(bias_effects)
    best_max_affected = best_group_value(affected_intersections_effects)
    best_gating = best_group_value(gating_effects)
    best_persistence = best_group_value(persistence_effects)
    best_decay = best_group_value(decay_effects)

    recommendation = None
    if best_config is not None:
        recommendation = (
            f"Start the next paired eval with {best_config['config_id']} "
            f"(bias={best_config['bias_strength']}, max_affected={best_config['max_intersections_affected']}, "
            f"gate={best_config['gating_mode']}, persistence={best_config['guidance_persistence_steps']}, "
            f"decay={best_config['enable_bias_decay']})."
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "comparison_scope": {
            "cities": list(args.cities),
            "scenarios": list(args.scenarios),
            "seeds": [int(seed) for seed in args.seeds],
            "episodes_per_seed": int(args.episodes_per_seed),
            "num_sweep_configs": int(len(sweep_configs)),
            "num_paired_rows": int(len(paired_rows)),
        },
        "rl_only_mean_total_return": rl_only_mean_total_return,
        "best_overall_config": best_config,
        "did_any_rl_llm_config_beat_rl_only": bool(configs_beating_rl_only),
        "closest_if_no_beat": None if configs_beating_rl_only else best_config,
        "top_5_configs": top_5,
        "parameter_effects": {
            "bias_strength": bias_effects,
            "max_intersections_affected": affected_intersections_effects,
            "gating_mode": gating_effects,
            "guidance_persistence_steps": persistence_effects,
            "enable_bias_decay": decay_effects,
        },
        "analysis_answers": {
            "which_config_was_best_overall": None if best_config is None else best_config["config_id"],
            "did_any_rl_llm_config_beat_rl_only": bool(configs_beating_rl_only),
            "did_weaker_bias_help": best_bias in {"0.025", "0.05"},
            "did_affecting_fewer_intersections_help": best_max_affected == "1",
            "did_gating_help": best_gating not in {None, "always_on"},
            "did_shorter_persistence_help": best_persistence == "5",
            "did_bias_decay_help": best_decay == "True",
        },
        "recommendation": recommendation,
    }


def group_rankings_by_parameter(
    ranking_rows: list[dict[str, Any]],
    parameter_name: str,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in ranking_rows:
        key = str(row[parameter_name])
        buckets.setdefault(key, []).append(row)
    grouped: list[dict[str, Any]] = []
    for key, rows in sorted(buckets.items(), key=lambda item: item[0]):
        grouped.append(
            {
                "value": key,
                "num_configs": int(len(rows)),
                "mean_return_delta_vs_rl_only": distribution_summary(
                    [safe_float(row.get("mean_return_delta_vs_rl_only")) for row in rows]
                )["mean"],
                "mean_throughput_delta_vs_rl_only": distribution_summary(
                    [safe_float(row.get("mean_throughput_delta_vs_rl_only")) for row in rows]
                )["mean"],
                "mean_avg_queue_delta_vs_rl_only": distribution_summary(
                    [safe_float(row.get("mean_avg_queue_delta_vs_rl_only")) for row in rows]
                )["mean"],
                "mean_avg_wait_delta_vs_rl_only": distribution_summary(
                    [safe_float(row.get("mean_avg_wait_delta_vs_rl_only")) for row in rows]
                )["mean"],
                "mean_percent_steps_with_active_guidance": distribution_summary(
                    [safe_float(row.get("mean_percent_steps_with_active_guidance")) for row in rows]
                )["mean"],
            }
        )
    grouped.sort(
        key=lambda item: (
            float(item["mean_return_delta_vs_rl_only"]),
            float(item["mean_throughput_delta_vs_rl_only"]),
            -float(item["mean_avg_queue_delta_vs_rl_only"]),
            -float(item["mean_avg_wait_delta_vs_rl_only"]),
        ),
        reverse=True,
    )
    return grouped


def best_group_value(grouped_rows: list[dict[str, Any]]) -> str | None:
    return grouped_rows[0]["value"] if grouped_rows else None


def build_config_payload(
    args: argparse.Namespace,
    env_config: EnvConfig,
    episode_plans: list[Any],
    sweep_configs: list[SweepConfigSpec],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "rl_checkpoint": str(args.rl_checkpoint),
        "llm_model_path": str(args.llm_model_path),
        "comparison_scope": {
            "num_episode_plans": int(len(episode_plans)),
            "cities": sorted({plan.city_id for plan in episode_plans}),
            "scenarios": sorted({plan.scenario for plan in episode_plans}),
            "seeds": sorted({int(plan.seed) for plan in episode_plans}),
            "episodes_per_seed": int(args.episodes_per_seed),
            "max_episode_seconds": args.max_episode_seconds,
            "total_runs": int(len(episode_plans) * (1 + len(sweep_configs))),
        },
        "episode_plans": [plan.to_dict() for plan in episode_plans],
        "sweep_configs": [config.to_dict() for config in sweep_configs],
        "influence_configs": {
            config.config_id: guidance_config_payload(config.to_influence_config())
            for config in sweep_configs
        },
        "repair_config": asdict(
            RepairConfig(
                allow_only_visible_candidates=args.allow_only_visible_candidates,
                max_target_intersections=args.max_target_intersections,
                fallback_on_empty_targets=args.fallback_on_empty_targets,
                fallback_mode=args.fallback_mode,
            )
        ),
        "env_config": env_config_to_payload(env_config),
        "save_step_metrics": bool(args.save_step_metrics),
        "save_guidance_traces": bool(args.save_guidance_traces),
    }


def format_float_token(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def gating_mode_token(value: str) -> str:
    return {
        "always_on": "always",
        "incident_or_spillback": "incident",
        "queue_threshold": "queue",
        "imbalance_threshold": "imbalance",
        "queue_or_imbalance": "queue_or_imb",
        "combined": "combined",
    }[value]


if __name__ == "__main__":
    main()
