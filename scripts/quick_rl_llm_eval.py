from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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
    run_episode,
    safe_float,
    try_write_parquet,
    write_csv_rows,
    write_json,
)
from training.cityflow_dataset import CityFlowDataset


DEFAULT_SEEDS: tuple[int, ...] = (7,)
PREFERRED_DEFAULT_CITIES: tuple[str, ...] = ("city_0001",)
PREFERRED_DEFAULT_SCENARIOS: tuple[str, ...] = ("normal",)
SCENARIO_ALIASES: dict[str, str] = {
    "rush": "morning_rush",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quick paired evaluation for rl_only vs rl_heuristic vs rl_llm using the "
            "best target_only_soft wrapper settings."
        )
    )
    parser.add_argument("--rl-checkpoint", required=True)
    parser.add_argument("--llm-model-path", required=True)
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument(
        "--max-episode-seconds",
        type=int,
        default=120,
        help="Short default horizon so the quick check stays under roughly 10-20 minutes.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="artifacts/quick_rl_llm_eval")
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

    city_ids = resolve_quick_cities(dataset=dataset, requested_cities=args.cities)
    scenario_specs = resolve_quick_scenario_specs(
        dataset=dataset,
        city_ids=city_ids,
        requested_scenarios=args.scenarios,
    )
    episode_plans = build_episode_plans(
        scenario_specs=scenario_specs,
        seeds=args.seeds,
        num_episodes=args.episodes_per_seed,
        seeded_config_root=seeded_config_root,
    )

    rl_policy = FixedRLPolicyAdapter(
        checkpoint_path=args.rl_checkpoint,
        device=args.device,
    )
    env_config = rl_policy.env_config or default_env_config()
    env_config = EnvConfig(
        simulator_interval=env_config.simulator_interval,
        decision_interval=env_config.decision_interval,
        min_green_time=env_config.min_green_time,
        thread_num=env_config.thread_num,
        max_episode_seconds=int(args.max_episode_seconds),
        observation=env_config.observation,
        reward=env_config.reward,
    )

    tuned_config = GuidanceInfluenceConfig(
        wrapper_mode="target_only_soft",
        bias_strength=0.025,
        target_only_bias_strength=0.025,
        corridor_bias_strength=0.0125,
        max_intersections_affected=2,
        guidance_refresh_steps=10,
        guidance_persistence_steps=5,
        max_guidance_duration=10,
        apply_global_bias=False,
        apply_target_only=True,
        gating_mode="queue_or_imbalance",
        min_avg_queue_for_guidance=150.0,
        min_queue_imbalance_for_guidance=20.0,
        require_incident_or_spillback=False,
        allow_guidance_in_normal_conditions=False,
        enable_bias_decay=False,
        bias_decay_schedule="linear",
        fallback_policy="no_op",
        log_guidance_debug=False,
    ).validate()

    controllers = build_controllers(
        args=args,
        rl_policy=rl_policy,
        tuned_config=tuned_config,
    )

    episode_rows: list[dict[str, Any]] = []
    rows_by_pair: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = {}
    total_runs = len(episode_plans) * len(controllers)
    progress = tqdm(total=total_runs, desc="Quick RL+LLM eval", unit="run")
    try:
        for plan in episode_plans:
            for mode_label, controller in controllers.items():
                progress.set_postfix_str(
                    f"mode={mode_label} city={plan.city_id} scenario={plan.scenario} seed={plan.seed}"
                )
                episode_row, _, _ = run_episode(
                    plan=plan,
                    mode_label=mode_label,
                    controller=controller,
                    env_config=env_config,
                    save_step_metrics=False,
                    save_guidance_traces=False,
                    show_step_progress=False,
                )
                episode_row = augment_episode_row(episode_row, tuned_config)
                episode_rows.append(episode_row)
                rows_by_pair.setdefault(plan.pairing_key(), {})[mode_label] = episode_row
                progress.update(1)
    finally:
        progress.close()

    paired_delta_rows = build_paired_delta_rows(rows_by_pair)
    summary_payload = build_summary_payload(
        episode_rows=episode_rows,
        paired_delta_rows=paired_delta_rows,
        tuned_config=tuned_config,
        args=args,
        scenario_specs=scenario_specs,
    )

    write_csv_rows(output_dir / "episode_metrics.csv", episode_rows)
    episode_parquet_written = try_write_parquet(output_dir / "episode_metrics.parquet", episode_rows)
    write_csv_rows(output_dir / "paired_deltas.csv", paired_delta_rows)
    try_write_parquet(output_dir / "paired_deltas.parquet", paired_delta_rows)
    write_json(output_dir / "summary.json", summary_payload)

    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    if not episode_parquet_written:
        print(
            "[warning] episode_metrics.parquet was not written because neither pyarrow nor pandas "
            "is available in the current Python environment."
        )


def resolve_quick_cities(
    dataset: CityFlowDataset,
    requested_cities: list[str] | None,
) -> list[str]:
    available = set(dataset.discover_cities())
    if requested_cities:
        selected = [city_id for city_id in requested_cities if city_id in available]
        if not selected:
            raise ValueError(f"None of the requested cities are available: {requested_cities}")
        return selected
    defaults = [city_id for city_id in PREFERRED_DEFAULT_CITIES if city_id in available]
    if defaults:
        return defaults[:1]
    discovered = sorted(available)
    if not discovered:
        raise ValueError("No generated cities were found under the generated-root.")
    return discovered[:1]


def resolve_quick_scenario_specs(
    dataset: CityFlowDataset,
    city_ids: list[str],
    requested_scenarios: list[str] | None,
) -> list[Any]:
    specs: list[Any] = []
    for city_id in city_ids:
        available_scenarios = set(dataset.scenarios_for_city(city_id))
        if requested_scenarios:
            desired = [
                SCENARIO_ALIASES.get(scenario_name, scenario_name)
                for scenario_name in requested_scenarios
            ]
        else:
            desired = [
                scenario_name
                for scenario_name in PREFERRED_DEFAULT_SCENARIOS
                if scenario_name in available_scenarios
            ][:2]
        selected = [scenario_name for scenario_name in desired if scenario_name in available_scenarios]
        if not selected:
            raise ValueError(
                f"No requested/default scenarios are available for city '{city_id}'. "
                f"Available scenarios: {sorted(available_scenarios)}"
            )
        for scenario_name in selected:
            specs.append(dataset.build_scenario_spec(city_id, scenario_name))
    if not specs:
        raise ValueError("No scenario specs were resolved for the quick evaluation.")
    return specs


def build_controllers(
    args: argparse.Namespace,
    rl_policy: FixedRLPolicyAdapter,
    tuned_config: GuidanceInfluenceConfig,
) -> dict[str, DistrictGuidedRLController]:
    heuristic_provider = HeuristicGuidanceProvider(
        config=HeuristicGuidanceConfig(
            max_target_intersections=args.max_target_intersections,
        )
    )
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
    llm_provider = LLMGuidanceProvider(
        inference=llm_inference,
        max_new_tokens=args.max_new_tokens,
    )

    def summary_builder() -> DistrictStateSummaryBuilder:
        return DistrictStateSummaryBuilder(
            top_k=3,
            candidate_limit=max(6, int(args.max_target_intersections)),
        )

    return {
        "rl_only": DistrictGuidedRLController(
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
                guidance_refresh_steps=tuned_config.guidance_refresh_steps,
                guidance_persistence_steps=1,
                max_guidance_duration=tuned_config.max_guidance_duration,
                fallback_policy="no_op",
                enable_bias_decay=False,
            ),
            heuristic_provider=None,
        ),
        "rl_heuristic": DistrictGuidedRLController(
            policy=rl_policy,
            mode_source="rl_heuristic",
            summary_builder=summary_builder(),
            guidance_provider=heuristic_provider,
            influence_config=tuned_config,
            heuristic_provider=heuristic_provider,
        ),
        "rl_llm": DistrictGuidedRLController(
            policy=rl_policy,
            mode_source="rl_llm",
            summary_builder=summary_builder(),
            guidance_provider=llm_provider,
            influence_config=tuned_config,
            heuristic_provider=heuristic_provider,
        ),
    }


def augment_episode_row(
    row: dict[str, Any],
    tuned_config: GuidanceInfluenceConfig,
) -> dict[str, Any]:
    payload = dict(row)
    payload.update(
        {
            "wrapper_mode": tuned_config.wrapper_mode if row["mode"] != "rl_only" else "no_op",
            "bias_strength": 0.0 if row["mode"] == "rl_only" else tuned_config.bias_strength,
            "target_only_bias_strength": 0.0
            if row["mode"] == "rl_only"
            else tuned_config.target_only_bias_strength,
            "corridor_bias_strength": 0.0
            if row["mode"] == "rl_only"
            else tuned_config.corridor_bias_strength,
            "max_intersections_affected": 0
            if row["mode"] == "rl_only"
            else tuned_config.max_intersections_affected,
            "gating_mode": "always_on" if row["mode"] == "rl_only" else tuned_config.gating_mode,
            "guidance_persistence_steps": 0
            if row["mode"] == "rl_only"
            else tuned_config.guidance_persistence_steps,
            "guidance_refresh_steps": 0
            if row["mode"] == "rl_only"
            else tuned_config.guidance_refresh_steps,
            "enable_bias_decay": False if row["mode"] == "rl_only" else tuned_config.enable_bias_decay,
            "min_avg_queue_for_guidance": 0.0
            if row["mode"] == "rl_only"
            else tuned_config.min_avg_queue_for_guidance,
            "min_queue_imbalance_for_guidance": 0.0
            if row["mode"] == "rl_only"
            else tuned_config.min_queue_imbalance_for_guidance,
        }
    )
    return payload


def build_paired_delta_rows(
    rows_by_pair: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    comparison_modes = ("rl_heuristic", "rl_llm")
    paired_rows: list[dict[str, Any]] = []
    for (city_id, scenario, seed, episode_id), mode_rows in sorted(rows_by_pair.items()):
        rl_only_row = mode_rows.get("rl_only")
        if rl_only_row is None:
            continue
        for comparison_mode in comparison_modes:
            other_row = mode_rows.get(comparison_mode)
            if other_row is None:
                continue
            paired_rows.append(
                {
                    "city_id": city_id,
                    "scenario": scenario,
                    "seed": int(seed),
                    "episode_id": int(episode_id),
                    "comparison": f"{comparison_mode}_vs_rl_only",
                    "mode": comparison_mode,
                    "total_return_delta": safe_float(other_row.get("total_return"))
                    - safe_float(rl_only_row.get("total_return")),
                    "avg_queue_delta": safe_float(other_row.get("avg_queue"))
                    - safe_float(rl_only_row.get("avg_queue")),
                    "avg_wait_delta": safe_float(other_row.get("avg_wait"))
                    - safe_float(rl_only_row.get("avg_wait")),
                    "throughput_delta": safe_float(other_row.get("throughput"))
                    - safe_float(rl_only_row.get("throughput")),
                    "travel_time_delta": safe_float(other_row.get("travel_time"))
                    - safe_float(rl_only_row.get("travel_time")),
                    "spillback_delta": safe_float(other_row.get("spillback_count"))
                    - safe_float(rl_only_row.get("spillback_count")),
                    "return_beats_rl_only": float(
                        safe_float(other_row.get("total_return"))
                        > safe_float(rl_only_row.get("total_return"))
                    ),
                }
            )
    return paired_rows


def build_summary_payload(
    episode_rows: list[dict[str, Any]],
    paired_delta_rows: list[dict[str, Any]],
    tuned_config: GuidanceInfluenceConfig,
    args: argparse.Namespace,
    scenario_specs: list[Any],
) -> dict[str, Any]:
    metrics_by_mode: dict[str, dict[str, float]] = {}
    for mode in ("rl_only", "rl_heuristic", "rl_llm"):
        mode_rows = [row for row in episode_rows if row["mode"] == mode]
        metrics_by_mode[mode] = {
            "mean_total_return": distribution_summary(
                [safe_float(row.get("total_return")) for row in mode_rows]
            )["mean"],
            "std_total_return": distribution_summary(
                [safe_float(row.get("total_return")) for row in mode_rows]
            )["std"],
            "mean_avg_queue": distribution_summary(
                [safe_float(row.get("avg_queue")) for row in mode_rows]
            )["mean"],
            "mean_avg_wait": distribution_summary(
                [safe_float(row.get("avg_wait")) for row in mode_rows]
            )["mean"],
            "mean_throughput": distribution_summary(
                [safe_float(row.get("throughput")) for row in mode_rows]
            )["mean"],
            "mean_travel_time": distribution_summary(
                [safe_float(row.get("travel_time")) for row in mode_rows]
            )["mean"],
            "mean_spillback_count": distribution_summary(
                [safe_float(row.get("spillback_count")) for row in mode_rows]
            )["mean"],
            "mean_percent_steps_with_active_guidance": distribution_summary(
                [safe_float(row.get("percent_steps_with_active_guidance")) for row in mode_rows]
            )["mean"],
            "mean_avg_num_affected_intersections": distribution_summary(
                [safe_float(row.get("avg_num_affected_intersections")) for row in mode_rows]
            )["mean"],
            "mean_fallback_used_count": distribution_summary(
                [safe_float(row.get("fallback_used_count")) for row in mode_rows]
            )["mean"],
            "mean_invalid_guidance_count": distribution_summary(
                [safe_float(row.get("invalid_guidance_count")) for row in mode_rows]
            )["mean"],
        }

    rl_only_metrics = metrics_by_mode["rl_only"]
    paired_summary = {
        comparison: {
            "mean_total_return_delta": distribution_summary(
                [safe_float(row.get("total_return_delta")) for row in paired_delta_rows if row["comparison"] == comparison]
            )["mean"],
            "std_total_return_delta": distribution_summary(
                [safe_float(row.get("total_return_delta")) for row in paired_delta_rows if row["comparison"] == comparison]
            )["std"],
            "mean_avg_queue_delta": distribution_summary(
                [safe_float(row.get("avg_queue_delta")) for row in paired_delta_rows if row["comparison"] == comparison]
            )["mean"],
            "mean_avg_wait_delta": distribution_summary(
                [safe_float(row.get("avg_wait_delta")) for row in paired_delta_rows if row["comparison"] == comparison]
            )["mean"],
            "mean_throughput_delta": distribution_summary(
                [safe_float(row.get("throughput_delta")) for row in paired_delta_rows if row["comparison"] == comparison]
            )["mean"],
            "beats_fraction": distribution_summary(
                [safe_float(row.get("return_beats_rl_only")) for row in paired_delta_rows if row["comparison"] == comparison]
            )["mean"],
        }
        for comparison in ("rl_heuristic_vs_rl_only", "rl_llm_vs_rl_only")
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": {
            "cities": sorted({spec.city_id for spec in scenario_specs}),
            "scenarios": sorted({spec.scenario_name for spec in scenario_specs}),
            "seeds": [int(seed) for seed in args.seeds],
            "episodes_per_seed": int(args.episodes_per_seed),
            "max_episode_seconds": int(args.max_episode_seconds),
            "total_runs": int(len(episode_rows)),
        },
        "wrapper_config": guidance_config_payload(tuned_config),
        "repair_config": asdict(
            RepairConfig(
                allow_only_visible_candidates=args.allow_only_visible_candidates,
                max_target_intersections=args.max_target_intersections,
                fallback_on_empty_targets=args.fallback_on_empty_targets,
                fallback_mode=args.fallback_mode,
            )
        ),
        "metrics_by_mode": metrics_by_mode,
        "paired_summary": paired_summary,
        "rl_only_mean_return": rl_only_metrics["mean_total_return"],
        "rl_heuristic_mean_return": metrics_by_mode["rl_heuristic"]["mean_total_return"],
        "rl_llm_mean_return": metrics_by_mode["rl_llm"]["mean_total_return"],
        "rl_heuristic_return_delta_vs_rl_only": (
            metrics_by_mode["rl_heuristic"]["mean_total_return"] - rl_only_metrics["mean_total_return"]
        ),
        "rl_llm_return_delta_vs_rl_only": (
            metrics_by_mode["rl_llm"]["mean_total_return"] - rl_only_metrics["mean_total_return"]
        ),
        "rl_heuristic_avg_queue_delta_vs_rl_only": (
            metrics_by_mode["rl_heuristic"]["mean_avg_queue"] - rl_only_metrics["mean_avg_queue"]
        ),
        "rl_llm_avg_queue_delta_vs_rl_only": (
            metrics_by_mode["rl_llm"]["mean_avg_queue"] - rl_only_metrics["mean_avg_queue"]
        ),
        "rl_heuristic_avg_wait_delta_vs_rl_only": (
            metrics_by_mode["rl_heuristic"]["mean_avg_wait"] - rl_only_metrics["mean_avg_wait"]
        ),
        "rl_llm_avg_wait_delta_vs_rl_only": (
            metrics_by_mode["rl_llm"]["mean_avg_wait"] - rl_only_metrics["mean_avg_wait"]
        ),
        "rl_heuristic_throughput_delta_vs_rl_only": (
            metrics_by_mode["rl_heuristic"]["mean_throughput"] - rl_only_metrics["mean_throughput"]
        ),
        "rl_llm_throughput_delta_vs_rl_only": (
            metrics_by_mode["rl_llm"]["mean_throughput"] - rl_only_metrics["mean_throughput"]
        ),
        "heuristic_beats_rl_fraction": paired_summary["rl_heuristic_vs_rl_only"]["beats_fraction"],
        "llm_beats_rl_fraction": paired_summary["rl_llm_vs_rl_only"]["beats_fraction"],
    }


if __name__ == "__main__":
    main()
