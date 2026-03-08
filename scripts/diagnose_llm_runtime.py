from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from district_llm.eval import load_rows
from district_llm.inference import DistrictLLMInference
from district_llm.prompting import (
    build_chat_messages,
    build_system_prompt,
    build_user_prompt,
    format_district_prompt,
    format_district_prompt_from_user_content,
    render_chat_prompt,
)
from district_llm.repair import (
    RepairConfig,
    parse_candidate_intersections_from_text,
)
from district_llm.rl_guidance_wrapper import FixedRLPolicyAdapter
from district_llm.schema import (
    DISTRICT_STRATEGIES,
    PHASE_BIASES,
    PRIORITY_CORRIDORS,
    CandidateIntersection,
    CongestedIntersection,
    DistrictAction,
    DistrictStateSummary,
)
from district_llm.summary_builder import DistrictStateSummaryBuilder
from env.observation_builder import ObservationConfig
from env.reward import RewardConfig
from env.traffic_env import EnvConfig
from training.cityflow_dataset import CityFlowDataset, ScenarioSpec
from training.train_local_policy import build_env


REQUIRED_ACTION_KEYS = {
    "strategy",
    "priority_corridor",
    "target_intersections",
    "phase_bias",
    "duration_steps",
}
SUMMARY_SCALAR_ORDER = [
    "city_id",
    "district_id",
    "district_type",
    "scenario",
    "scenario_type",
    "decision_step",
    "sim_time",
    "intersection_count",
    "avg_queue",
    "max_queue",
    "total_queue",
    "avg_wait",
    "max_wait",
    "total_wait",
    "avg_outgoing_load",
    "max_outgoing_load",
    "total_outgoing_load",
    "recent_throughput",
    "queue_change",
    "wait_change",
    "throughput_change",
    "ns_queue",
    "ew_queue",
    "ns_wait",
    "ew_wait",
    "dominant_flow",
    "boundary_queue_total",
    "boundary_wait_total",
    "spillback_risk",
    "incident_flag",
    "construction_flag",
    "overload_flag",
    "event_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why district-LLM runtime guidance fails even when offline validation looks strong."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rl-checkpoint", required=True)
    parser.add_argument("--val-jsonl", default="data/district_llm_dataset_v3/val.jsonl")
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--max-diagnostic-calls", type=int, default=20)
    parser.add_argument("--max-offline-examples", type=int, default=20)
    parser.add_argument("--max-episode-seconds", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="artifacts/llm_runtime_diagnosis")
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

    repair_config = RepairConfig(
        allow_only_visible_candidates=args.allow_only_visible_candidates,
        max_target_intersections=args.max_target_intersections,
        fallback_on_empty_targets=args.fallback_on_empty_targets,
        fallback_mode=args.fallback_mode,
    )
    inference = DistrictLLMInference(
        model_name_or_path=args.model_path,
        device=args.device,
        repair_config=repair_config,
    )

    runtime_rows = collect_runtime_rows(args=args, inference=inference)
    offline_rows = collect_offline_rows(args=args, inference=inference)
    all_rows = runtime_rows + offline_rows

    failure_examples = [
        flatten_failure_example(row, prompt_style_key)
        for row in all_rows
        for prompt_style_key in ("runtime_flat", "training_chat")
        if row[prompt_style_key]["wrapper_would_fallback"]
    ]
    summary_report = build_summary_report(
        args=args,
        inference=inference,
        runtime_rows=runtime_rows,
        offline_rows=offline_rows,
    )
    prompt_comparison = render_prompt_comparison(
        runtime_rows=runtime_rows,
        offline_rows=offline_rows,
        summary_report=summary_report,
    )

    write_jsonl(output_dir / "diagnostic_rows.jsonl", all_rows)
    write_json(output_dir / "summary_report.json", summary_report)
    write_text(output_dir / "prompt_comparison.md", prompt_comparison)
    write_jsonl(output_dir / "validator_failure_examples.jsonl", failure_examples)

    print(json.dumps(summary_report, indent=2, sort_keys=True))


def collect_runtime_rows(
    args: argparse.Namespace,
    inference: DistrictLLMInference,
) -> list[dict[str, Any]]:
    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    dataset.generate_default_splits()
    scenario_specs = resolve_scenario_specs(dataset=dataset, args=args)

    policy = FixedRLPolicyAdapter(
        checkpoint_path=args.rl_checkpoint,
        device=args.device,
    )
    env_config = policy.env_config or default_env_config()
    env_config = EnvConfig(
        simulator_interval=env_config.simulator_interval,
        decision_interval=env_config.decision_interval,
        min_green_time=env_config.min_green_time,
        thread_num=env_config.thread_num,
        max_episode_seconds=int(args.max_episode_seconds),
        observation=env_config.observation,
        reward=env_config.reward,
    )

    summary_builder = DistrictStateSummaryBuilder(
        top_k=3,
        candidate_limit=max(6, args.max_target_intersections),
    )
    rows: list[dict[str, Any]] = []

    for scenario_spec in scenario_specs:
        if len(rows) >= args.max_diagnostic_calls:
            break
        env = build_env(env_config, scenario_spec)
        observation_batch = env.reset()
        summary_builder.reset()
        done = False
        while not done and len(rows) < args.max_diagnostic_calls:
            summaries = summary_builder.build_all(env, observation_batch)
            for district_id in sorted(summaries):
                if len(rows) >= args.max_diagnostic_calls:
                    break
                summary = summaries[district_id]
                rows.append(
                    diagnose_summary_call(
                        inference=inference,
                        summary=summary,
                        source="runtime_live",
                        city_id=scenario_spec.city_id,
                        scenario=scenario_spec.scenario_name,
                        district_id=district_id,
                        decision_step=int(summary.decision_step),
                        wrapper_mode="diagnose_llm_runtime",
                        max_new_tokens=args.max_new_tokens,
                    )
                )
            if len(rows) >= args.max_diagnostic_calls:
                break
            actions = policy.decide(observation_batch).actions
            observation_batch, _, done, _ = env.step(actions)
    return rows


def collect_offline_rows(
    args: argparse.Namespace,
    inference: DistrictLLMInference,
) -> list[dict[str, Any]]:
    raw_rows = load_rows(args.val_jsonl, max_examples=args.max_offline_examples)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        training_messages = row["messages"][:2]
        summary_text = row["messages"][1]["content"]
        summary, summary_parse = parse_summary_text(summary_text)
        rows.append(
            diagnose_summary_call(
                inference=inference,
                summary=summary,
                source="offline_validation_runtime_codepath",
                city_id=str(row.get("city_id", summary.city_id)),
                scenario=str(row.get("scenario", summary.scenario_name)),
                district_id=str(row.get("district_id", summary.district_id)),
                decision_step=int(summary.decision_step),
                wrapper_mode="diagnose_llm_runtime",
                max_new_tokens=args.max_new_tokens,
                training_messages=training_messages,
                ground_truth_payload=json.loads(row["messages"][2]["content"]),
                original_user_prompt=summary_text,
                summary_parse=summary_parse,
                example_index=index,
            )
        )
    return rows


def diagnose_summary_call(
    inference: DistrictLLMInference,
    summary: DistrictStateSummary,
    source: str,
    city_id: str,
    scenario: str,
    district_id: str,
    decision_step: int,
    wrapper_mode: str,
    max_new_tokens: int,
    training_messages: list[dict[str, str]] | None = None,
    ground_truth_payload: dict[str, Any] | None = None,
    original_user_prompt: str | None = None,
    summary_parse: dict[str, Any] | None = None,
    example_index: int | None = None,
) -> dict[str, Any]:
    training_messages = training_messages or build_chat_messages(
        summary,
        max_target_intersections=inference.repair_config.max_target_intersections,
        allow_only_visible_candidates=inference.repair_config.allow_only_visible_candidates,
    )
    runtime_user_prompt = original_user_prompt or build_user_prompt(summary)
    runtime_flat_prompt = format_district_prompt(
        summary,
        max_target_intersections=inference.repair_config.max_target_intersections,
        allow_only_visible_candidates=inference.repair_config.allow_only_visible_candidates,
    )
    training_chat_prompt = render_chat_prompt(
        training_messages,
        tokenizer=inference.tokenizer,
        add_generation_prompt=True,
    )
    runtime_flat_from_user_prompt = format_district_prompt_from_user_content(
        runtime_user_prompt,
        max_target_intersections=inference.repair_config.max_target_intersections,
        allow_only_visible_candidates=inference.repair_config.allow_only_visible_candidates,
    )

    runtime_flat = run_prompt_diagnostic(
        inference=inference,
        prompt_text=runtime_flat_prompt,
        summary=summary,
        max_new_tokens=max_new_tokens,
        prompt_style="runtime_flat",
    )
    training_chat = run_prompt_diagnostic(
        inference=inference,
        prompt_text=training_chat_prompt,
        summary=summary,
        max_new_tokens=max_new_tokens,
        prompt_style="training_chat",
    )

    training_system_prompt = training_messages[0]["content"] if training_messages else build_system_prompt(
        max_target_intersections=inference.repair_config.max_target_intersections,
        allow_only_visible_candidates=inference.repair_config.allow_only_visible_candidates,
    )
    training_user_prompt = training_messages[1]["content"] if len(training_messages) > 1 else runtime_user_prompt

    prompt_compare = compare_prompt_shapes(
        training_system_prompt=training_system_prompt,
        training_user_prompt=training_user_prompt,
        runtime_flat_prompt=runtime_flat_prompt,
        runtime_flat_from_user_prompt=runtime_flat_from_user_prompt,
        training_chat_prompt=training_chat_prompt,
    )

    row = {
        "source": source,
        "example_index": example_index,
        "city_id": city_id,
        "scenario": scenario,
        "district_id": district_id,
        "decision_step": int(decision_step),
        "wrapper_mode": wrapper_mode,
        "training_system_prompt": training_system_prompt,
        "training_user_prompt": training_user_prompt,
        "runtime_flat_prompt": runtime_flat_prompt,
        "runtime_flat_prompt_from_user_prompt": runtime_flat_from_user_prompt,
        "training_chat_prompt": training_chat_prompt,
        "prompt_comparison": prompt_compare,
        "summary_features": summary_features(runtime_user_prompt, summary_parse=summary_parse),
        "summary_state": summary.to_dict(),
        "runtime_flat": runtime_flat,
        "training_chat": training_chat,
        "ground_truth_payload": ground_truth_payload,
    }
    return row


def run_prompt_diagnostic(
    inference: DistrictLLMInference,
    prompt_text: str,
    summary: DistrictStateSummary,
    max_new_tokens: int,
    prompt_style: str,
) -> dict[str, Any]:
    raw_text = inference.generate_raw(prompt=prompt_text, max_new_tokens=max_new_tokens)
    action, repair_report, parsed_payload, json_valid, schema_valid_before_repair = inference.parse_action(
        raw_text,
        summary=summary,
    )
    prompt_token_length = token_length(inference, prompt_text)
    output_token_length = token_length(inference, raw_text)
    diagnostics = analyze_generation_result(
        raw_text=raw_text,
        parsed_payload=parsed_payload,
        summary=summary,
        repair_report=repair_report,
        max_new_tokens=max_new_tokens,
        output_token_length=output_token_length,
        json_valid=json_valid,
        schema_valid_before_repair=schema_valid_before_repair,
    )
    return {
        "prompt_style": prompt_style,
        "prompt_token_length": prompt_token_length,
        "output_token_length": output_token_length,
        "prompt_near_model_limit": prompt_near_model_limit(inference, prompt_token_length),
        "raw_text": raw_text,
        "extracted_json_text": diagnostics["extracted_json_text"],
        "parsed_payload_before_repair": parsed_payload,
        "action_after_repair": action.to_dict(),
        "repair_report": repair_report.to_dict(),
        "json_valid": bool(json_valid),
        "schema_valid_before_repair": bool(schema_valid_before_repair),
        "wrapper_would_fallback": bool(diagnostics["wrapper_would_fallback"]),
        "failure_reasons": diagnostics["failure_reasons"],
        "candidate_diagnostics": diagnostics["candidate_diagnostics"],
        "possible_truncation": bool(diagnostics["possible_truncation"]),
    }


def analyze_generation_result(
    raw_text: str,
    parsed_payload: dict[str, Any] | None,
    summary: DistrictStateSummary,
    repair_report,
    max_new_tokens: int,
    output_token_length: int | None,
    json_valid: bool,
    schema_valid_before_repair: bool,
) -> dict[str, Any]:
    failure_reasons: list[str] = []
    extracted_json_text, prefix_text, suffix_text = extract_json_details(raw_text)
    if "```" in raw_text:
        failure_reasons.append("markdown_code_fence_present")
    if prefix_text.strip():
        failure_reasons.append("extra_prefix_text")
    if suffix_text.strip():
        failure_reasons.append("extra_suffix_text")
    if extracted_json_text is None:
        failure_reasons.append("no_json_object_found")
    if not json_valid:
        failure_reasons.append("json_parse_error")

    raw_targets = []
    if parsed_payload is None:
        parsed_payload = None
    else:
        missing_keys = sorted(REQUIRED_ACTION_KEYS - set(parsed_payload))
        extra_keys = sorted(set(parsed_payload) - REQUIRED_ACTION_KEYS)
        if missing_keys:
            failure_reasons.append("missing_required_field")
        if extra_keys:
            failure_reasons.append("extra_field_present")
        strategy = parsed_payload.get("strategy")
        if strategy not in DISTRICT_STRATEGIES:
            failure_reasons.append("unknown_strategy")
        priority_corridor = parsed_payload.get("priority_corridor")
        if priority_corridor is not None and priority_corridor not in PRIORITY_CORRIDORS:
            failure_reasons.append("unknown_priority_corridor")
        phase_bias = parsed_payload.get("phase_bias")
        if phase_bias not in PHASE_BIASES:
            failure_reasons.append("unknown_phase_bias")
        duration_steps = parsed_payload.get("duration_steps")
        if not isinstance(duration_steps, int):
            failure_reasons.append("invalid_duration_type")
        elif not 1 <= duration_steps <= 20:
            failure_reasons.append("invalid_duration_range")

        raw_target_payload = parsed_payload.get("target_intersections", [])
        if isinstance(raw_target_payload, list):
            raw_targets = [str(item) for item in raw_target_payload]
        elif isinstance(raw_target_payload, str):
            failure_reasons.append("target_intersections_not_json_array")
            raw_targets = [raw_target_payload]
        else:
            failure_reasons.append("target_intersections_wrong_type")

    if not schema_valid_before_repair:
        failure_reasons.append("schema_validation_failed")
    if raw_text.strip() and not raw_text.rstrip().endswith("}"):
        failure_reasons.append("output_does_not_end_with_json")
    if output_token_length is not None and output_token_length >= max_new_tokens:
        failure_reasons.append("possible_generation_truncation")

    candidate_ids = set(summary.candidate_ids())
    candidate_diagnostics = []
    for target in raw_targets:
        visible = target in candidate_ids
        candidate_diagnostics.append(
            {
                "target_intersection": target,
                "visible_candidate": visible,
                "valid_id_format": target.startswith("i_"),
            }
        )
        if not target.startswith("i_"):
            failure_reasons.append("invalid_target_id_format")
        if candidate_ids and not visible:
            failure_reasons.append("candidate_intersections_constraint_violation")

    if raw_targets == []:
        failure_reasons.append("empty_target_intersections")
    if repair_report.invalid_ids_removed:
        failure_reasons.append("repair_removed_invalid_ids")
    if repair_report.non_visible_ids_removed:
        failure_reasons.append("repair_removed_non_visible_ids")
    if repair_report.empty_after_filtering:
        failure_reasons.append("repair_emptied_targets")
    if repair_report.fallback_used:
        failure_reasons.append(f"repair_used_fallback:{repair_report.fallback_mode}")

    wrapper_would_fallback = (
        not json_valid
        or not schema_valid_before_repair
        or bool(repair_report.fallback_used)
        or bool(repair_report.empty_after_filtering)
    )
    return {
        "extracted_json_text": extracted_json_text,
        "failure_reasons": sorted(set(failure_reasons)),
        "candidate_diagnostics": candidate_diagnostics,
        "possible_truncation": bool(output_token_length is not None and output_token_length >= max_new_tokens),
        "wrapper_would_fallback": wrapper_would_fallback,
    }


def extract_json_details(raw_text: str) -> tuple[str | None, str, str]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, raw_text, ""
    return raw_text[start : end + 1], raw_text[:start], raw_text[end + 1 :]


def parse_summary_text(summary_text: str) -> tuple[DistrictStateSummary, dict[str, Any]]:
    text = summary_text.strip()
    if text.startswith("### DISTRICT STATE"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]

    payload: dict[str, Any] = {}
    top_lines: list[str] = []
    candidate_lines: list[str] = []
    observed_order: list[str] = []
    section = "scalars"
    for line in lines:
        if line == "top_congested_intersections:":
            section = "top"
            continue
        if line == "candidate_intersections:":
            section = "candidate"
            continue
        if line.startswith("- "):
            if section == "top":
                top_lines.append(line)
            elif section == "candidate":
                candidate_lines.append(line)
            continue
        if section != "scalars" or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        observed_order.append(key)
        payload[key] = parse_summary_scalar(key, value)

    payload["scenario_name"] = payload.pop("scenario", payload.get("scenario_name", ""))
    payload["top_congested_intersections"] = [parse_top_congested_line(line) for line in top_lines if line != "- none"]
    candidate_text = "candidate_intersections:\n" + "\n".join(candidate_lines or ["- none"])
    payload["candidate_intersections"] = parse_candidate_intersections_from_text(candidate_text)
    summary = DistrictStateSummary.from_dict(payload)
    return summary, {
        "observed_field_order": observed_order,
        "missing_scalar_fields": [key for key in SUMMARY_SCALAR_ORDER if key not in observed_order],
        "extra_scalar_fields": [key for key in observed_order if key not in SUMMARY_SCALAR_ORDER],
        "top_congested_count": len(payload["top_congested_intersections"]),
        "candidate_intersections_count": len(payload["candidate_intersections"]),
        "summary_text_length": len(summary_text),
        "line_count": len(lines),
    }


def parse_summary_scalar(key: str, value: str) -> Any:
    if key in {
        "decision_step",
        "sim_time",
        "intersection_count",
    }:
        return int(value)
    if key in {
        "spillback_risk",
        "incident_flag",
        "construction_flag",
        "overload_flag",
        "event_flag",
    }:
        return bool(int(value))
    if key in {
        "city_id",
        "district_id",
        "district_type",
        "scenario",
        "scenario_type",
        "dominant_flow",
    }:
        return value
    return float(value)


def parse_top_congested_line(line: str) -> dict[str, Any]:
    tokens = line[2:].split()
    payload: dict[str, Any] = {
        "intersection_id": tokens[0],
        "queue_total": 0.0,
        "wait_total": 0.0,
        "outgoing_load": 0.0,
        "current_phase": 0,
        "is_boundary": False,
    }
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "q":
            payload["queue_total"] = float(value)
        elif key == "w":
            payload["wait_total"] = float(value)
        elif key == "out":
            payload["outgoing_load"] = float(value)
        elif key == "phase":
            payload["current_phase"] = int(value)
        elif key == "boundary":
            payload["is_boundary"] = value == "1"
    CongestedIntersection(**payload)
    return payload


def summary_features(summary_text: str, summary_parse: dict[str, Any] | None = None) -> dict[str, Any]:
    summary_parse = summary_parse or {}
    lines = [line.rstrip() for line in summary_text.splitlines() if line.strip()]
    field_order = []
    for line in lines:
        if ": " in line and not line.startswith("- "):
            field_order.append(line.split(": ", 1)[0])
        elif line.endswith(":") and not line.startswith("- "):
            field_order.append(line[:-1])
    return {
        "summary_length_chars": len(summary_text),
        "summary_line_count": len(lines),
        "field_order": field_order,
        "field_count": len(field_order),
        "has_candidate_intersections": "candidate_intersections:" in summary_text,
        "has_top_congested_intersections": "top_congested_intersections:" in summary_text,
        **summary_parse,
    }


def compare_prompt_shapes(
    training_system_prompt: str,
    training_user_prompt: str,
    runtime_flat_prompt: str,
    runtime_flat_from_user_prompt: str,
    training_chat_prompt: str,
) -> dict[str, Any]:
    differences = []
    if training_system_prompt and "You are a district traffic coordinator" in training_system_prompt:
        differences.append("training uses an explicit system prompt with JSON rules")
    if runtime_flat_prompt.startswith("### DISTRICT ACTION SCHEMA"):
        differences.append("runtime flat prompt injects schema text into the single prompt body")
    if training_user_prompt.startswith("### DISTRICT STATE"):
        differences.append("training user message contains only district state")
    if "### DECISION" in runtime_flat_prompt:
        differences.append("runtime flat prompt appends an explicit decision header")
    if "### DISTRICT ACTION SCHEMA" not in training_user_prompt:
        differences.append("training user message omits the schema header entirely")
    if runtime_flat_prompt != runtime_flat_from_user_prompt:
        differences.append("runtime flat prompt reconstructed from summary object differs from user prompt reconstruction")
    return {
        "differences": differences,
        "runtime_has_system_role": False,
        "training_has_system_role": True,
        "runtime_has_schema_header": "### DISTRICT ACTION SCHEMA" in runtime_flat_prompt,
        "training_user_has_schema_header": "### DISTRICT ACTION SCHEMA" in training_user_prompt,
        "runtime_has_decision_header": "### DECISION" in runtime_flat_prompt,
        "training_chat_prompt_startswith_system": training_chat_prompt.startswith("system:") or training_chat_prompt.startswith("<"),
    }


def aggregate_prompt_results(rows: list[dict[str, Any]], prompt_style_key: str) -> dict[str, Any]:
    if not rows:
        return {}
    prompt_rows = [row[prompt_style_key] for row in rows]
    failure_counter = Counter(
        reason
        for row in prompt_rows
        for reason in row["failure_reasons"]
    )
    candidate_rows = [row["candidate_diagnostics"] for row in prompt_rows]
    target_records = [record for records in candidate_rows for record in records]
    repair_reports = [row["repair_report"] for row in prompt_rows]
    return {
        "num_examples": len(prompt_rows),
        "json_valid_rate": safe_ratio(sum(int(row["json_valid"]) for row in prompt_rows), len(prompt_rows)),
        "schema_valid_before_repair_rate": safe_ratio(
            sum(int(row["schema_valid_before_repair"]) for row in prompt_rows),
            len(prompt_rows),
        ),
        "wrapper_would_fallback_rate": safe_ratio(
            sum(int(row["wrapper_would_fallback"]) for row in prompt_rows),
            len(prompt_rows),
        ),
        "repair_fallback_rate": safe_ratio(
            sum(int(report["fallback_used"]) for report in repair_reports),
            len(repair_reports),
        ),
        "repair_changed_target_list_rate": safe_ratio(
            sum(
                int(report["raw_targets"] != report["repaired_targets"])
                for report in repair_reports
            ),
            len(repair_reports),
        ),
        "repair_emptied_targets_rate": safe_ratio(
            sum(int(report["empty_after_filtering"]) for report in repair_reports),
            len(repair_reports),
        ),
        "mean_prompt_token_length": average([row["prompt_token_length"] for row in prompt_rows]),
        "mean_output_token_length": average([row["output_token_length"] for row in prompt_rows]),
        "possible_truncation_rate": safe_ratio(
            sum(int(row["possible_truncation"]) for row in prompt_rows),
            len(prompt_rows),
        ),
        "top_failure_reasons": dict(failure_counter.most_common(20)),
        "targets_outside_visible_candidate_rate": safe_ratio(
            sum(int(not record["visible_candidate"]) for record in target_records),
            len(target_records),
        ),
        "invalid_target_id_format_rate": safe_ratio(
            sum(int(not record["valid_id_format"]) for record in target_records),
            len(target_records),
        ),
    }


def aggregate_summary_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    features = [row["summary_features"] for row in rows]
    field_order_signatures = Counter(tuple(item["field_order"]) for item in features)
    return {
        "num_examples": len(features),
        "mean_summary_length_chars": average([item["summary_length_chars"] for item in features]),
        "median_summary_length_chars": median_or_zero([item["summary_length_chars"] for item in features]),
        "mean_candidate_intersections_count": average(
            [item.get("candidate_intersections_count", 0) for item in features]
        ),
        "mean_top_congested_count": average(
            [item.get("top_congested_count", 0) for item in features]
        ),
        "field_order_signatures": {
            "most_common": [
                {
                    "count": count,
                    "field_order": list(signature),
                }
                for signature, count in field_order_signatures.most_common(5)
            ]
        },
    }


def build_summary_report(
    args: argparse.Namespace,
    inference: DistrictLLMInference,
    runtime_rows: list[dict[str, Any]],
    offline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime_flat_runtime = aggregate_prompt_results(runtime_rows, "runtime_flat")
    training_chat_runtime = aggregate_prompt_results(runtime_rows, "training_chat")
    runtime_flat_offline = aggregate_prompt_results(offline_rows, "runtime_flat")
    training_chat_offline = aggregate_prompt_results(offline_rows, "training_chat")

    root_causes = rank_root_causes(
        runtime_flat_runtime=runtime_flat_runtime,
        training_chat_runtime=training_chat_runtime,
        runtime_flat_offline=runtime_flat_offline,
        training_chat_offline=training_chat_offline,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "rl_checkpoint": args.rl_checkpoint,
        "repair_config": asdict(inference.repair_config),
        "generation_settings": {
            "runtime_inference": {
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": False,
                "prompt_style": "flat single prompt from format_district_prompt",
            },
            "offline_eval_style": {
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": False,
                "prompt_style": "chat messages rendered via build_generation_prompt",
            },
        },
        "runtime_live": {
            "summary_distribution": aggregate_summary_features(runtime_rows),
            "runtime_flat": runtime_flat_runtime,
            "training_chat": training_chat_runtime,
        },
        "offline_validation_runtime_codepath": {
            "summary_distribution": aggregate_summary_features(offline_rows),
            "runtime_flat": runtime_flat_offline,
            "training_chat": training_chat_offline,
        },
        "key_answers": build_key_answers(
            runtime_flat_runtime=runtime_flat_runtime,
            training_chat_runtime=training_chat_runtime,
            runtime_flat_offline=runtime_flat_offline,
            training_chat_offline=training_chat_offline,
        ),
        "likely_root_causes_ranked": root_causes,
    }


def build_key_answers(
    runtime_flat_runtime: dict[str, Any],
    training_chat_runtime: dict[str, Any],
    runtime_flat_offline: dict[str, Any],
    training_chat_offline: dict[str, Any],
) -> dict[str, Any]:
    return {
        "runtime_prompt_vs_training_prompt": (
            "different"
            if runtime_flat_offline.get("wrapper_would_fallback_rate", 0.0)
            != training_chat_offline.get("wrapper_would_fallback_rate", 0.0)
            else "similar"
        ),
        "runtime_summary_structure_vs_training_distribution": (
            "requires inspection of summary_distribution stats and prompt comparison"
        ),
        "raw_outputs_malformed_or_rejected": {
            "runtime_flat_runtime_fallback_rate": runtime_flat_runtime.get("wrapper_would_fallback_rate"),
            "runtime_flat_runtime_json_valid_rate": runtime_flat_runtime.get("json_valid_rate"),
            "runtime_flat_runtime_schema_valid_rate": runtime_flat_runtime.get("schema_valid_before_repair_rate"),
        },
        "candidate_constraints_main_problem": bool(
            runtime_flat_runtime.get("targets_outside_visible_candidate_rate", 0.0) > 0.2
            or runtime_flat_offline.get("targets_outside_visible_candidate_rate", 0.0) > 0.2
        ),
        "truncation_happening": bool(
            runtime_flat_runtime.get("possible_truncation_rate", 0.0) > 0.05
            or runtime_flat_offline.get("possible_truncation_rate", 0.0) > 0.05
        ),
        "runtime_codepath_succeeds_on_heldout_validation": bool(
            runtime_flat_offline.get("wrapper_would_fallback_rate", 1.0) < 0.2
        ),
    }


def rank_root_causes(
    runtime_flat_runtime: dict[str, Any],
    training_chat_runtime: dict[str, Any],
    runtime_flat_offline: dict[str, Any],
    training_chat_offline: dict[str, Any],
) -> list[dict[str, Any]]:
    causes: list[dict[str, Any]] = []

    prompt_gap = (
        runtime_flat_offline.get("wrapper_would_fallback_rate", 0.0)
        - training_chat_offline.get("wrapper_would_fallback_rate", 0.0)
    )
    causes.append(
        {
            "cause": "prompt_mismatch_between_runtime_and_training_offline_chat_path",
            "score": float(prompt_gap),
            "evidence": {
                "offline_runtime_flat_fallback_rate": runtime_flat_offline.get("wrapper_would_fallback_rate"),
                "offline_training_chat_fallback_rate": training_chat_offline.get("wrapper_would_fallback_rate"),
            },
        }
    )

    runtime_summary_gap = (
        training_chat_runtime.get("wrapper_would_fallback_rate", 0.0)
        - training_chat_offline.get("wrapper_would_fallback_rate", 0.0)
    )
    causes.append(
        {
            "cause": "runtime_summary_distribution_shift",
            "score": float(runtime_summary_gap),
            "evidence": {
                "runtime_training_chat_fallback_rate": training_chat_runtime.get("wrapper_would_fallback_rate"),
                "offline_training_chat_fallback_rate": training_chat_offline.get("wrapper_would_fallback_rate"),
            },
        }
    )

    candidate_score = max(
        runtime_flat_runtime.get("targets_outside_visible_candidate_rate", 0.0),
        runtime_flat_offline.get("targets_outside_visible_candidate_rate", 0.0),
        runtime_flat_runtime.get("repair_emptied_targets_rate", 0.0),
        runtime_flat_offline.get("repair_emptied_targets_rate", 0.0),
    )
    causes.append(
        {
            "cause": "candidate_intersections_or_visible_target_constraint_mismatch",
            "score": float(candidate_score),
            "evidence": {
                "runtime_targets_outside_visible_rate": runtime_flat_runtime.get("targets_outside_visible_candidate_rate"),
                "offline_targets_outside_visible_rate": runtime_flat_offline.get("targets_outside_visible_candidate_rate"),
                "runtime_repair_emptied_targets_rate": runtime_flat_runtime.get("repair_emptied_targets_rate"),
            },
        }
    )

    validator_score = max(
        runtime_flat_runtime.get("wrapper_would_fallback_rate", 0.0)
        - runtime_flat_runtime.get("json_valid_rate", 0.0),
        runtime_flat_offline.get("wrapper_would_fallback_rate", 0.0)
        - runtime_flat_offline.get("json_valid_rate", 0.0),
    )
    causes.append(
        {
            "cause": "validator_or_repair_stricter_than_raw_generation_quality",
            "score": float(validator_score),
            "evidence": {
                "runtime_json_valid_rate": runtime_flat_runtime.get("json_valid_rate"),
                "runtime_wrapper_fallback_rate": runtime_flat_runtime.get("wrapper_would_fallback_rate"),
                "offline_json_valid_rate": runtime_flat_offline.get("json_valid_rate"),
                "offline_wrapper_fallback_rate": runtime_flat_offline.get("wrapper_would_fallback_rate"),
            },
        }
    )

    truncation_score = max(
        runtime_flat_runtime.get("possible_truncation_rate", 0.0),
        runtime_flat_offline.get("possible_truncation_rate", 0.0),
    )
    causes.append(
        {
            "cause": "generation_truncation",
            "score": float(truncation_score),
            "evidence": {
                "runtime_possible_truncation_rate": runtime_flat_runtime.get("possible_truncation_rate"),
                "offline_possible_truncation_rate": runtime_flat_offline.get("possible_truncation_rate"),
            },
        }
    )

    causes.sort(key=lambda item: item["score"], reverse=True)
    return causes


def render_prompt_comparison(
    runtime_rows: list[dict[str, Any]],
    offline_rows: list[dict[str, Any]],
    summary_report: dict[str, Any],
) -> str:
    runtime_example = runtime_rows[0] if runtime_rows else None
    offline_example = offline_rows[0] if offline_rows else None
    lines = [
        "# Runtime Prompt Diagnosis",
        "",
        "## Key Finding",
        "",
        "Training/offline evaluation uses a chat-style prompt with separate `system` and `user` messages.",
        "This report compares the chat-style prompt path against the older flattened prompt path.",
        "",
        "## Aggregate Answers",
        "",
        "```json",
        json.dumps(summary_report.get("key_answers", {}), indent=2, sort_keys=True),
        "```",
        "",
    ]
    if offline_example is not None:
        lines.extend(
            [
                "## Representative Offline Validation Example",
                "",
                "### Training System Prompt",
                "",
                "```text",
                offline_example["training_system_prompt"],
                "```",
                "",
                "### Training User Prompt",
                "",
                "```text",
                offline_example["training_user_prompt"],
                "```",
                "",
                "### Runtime Flat Prompt",
                "",
                "```text",
                offline_example["runtime_flat_prompt"],
                "```",
                "",
                "### Training Chat Rendered Prompt",
                "",
                "```text",
                offline_example["training_chat_prompt"],
                "```",
                "",
                "### Prompt Diff",
                "",
                "```diff",
                *list(
                    difflib.unified_diff(
                        offline_example["training_chat_prompt"].splitlines(),
                        offline_example["runtime_flat_prompt"].splitlines(),
                        fromfile="training_chat_prompt",
                        tofile="runtime_flat_prompt",
                        lineterm="",
                    )
                ),
                "```",
                "",
            ]
        )
    if runtime_example is not None:
        lines.extend(
            [
                "## Representative Runtime Summary Example",
                "",
                "### Runtime Flat Output",
                "",
                "```json",
                json.dumps(runtime_example["runtime_flat"], indent=2, sort_keys=True),
                "```",
                "",
                "### Training Chat Output On Same Summary",
                "",
                "```json",
                json.dumps(runtime_example["training_chat"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def flatten_failure_example(row: dict[str, Any], prompt_style_key: str) -> dict[str, Any]:
    payload = row[prompt_style_key]
    return {
        "source": row["source"],
        "prompt_style": prompt_style_key,
        "city_id": row["city_id"],
        "scenario": row["scenario"],
        "district_id": row["district_id"],
        "decision_step": row["decision_step"],
        "failure_reasons": payload["failure_reasons"],
        "json_valid": payload["json_valid"],
        "schema_valid_before_repair": payload["schema_valid_before_repair"],
        "wrapper_would_fallback": payload["wrapper_would_fallback"],
        "repair_report": payload["repair_report"],
        "raw_text": payload["raw_text"],
        "parsed_payload_before_repair": payload["parsed_payload_before_repair"],
        "action_after_repair": payload["action_after_repair"],
        "candidate_diagnostics": payload["candidate_diagnostics"],
        "prompt_text": row["runtime_flat_prompt"] if prompt_style_key == "runtime_flat" else row["training_chat_prompt"],
    }


def resolve_scenario_specs(dataset: CityFlowDataset, args: argparse.Namespace) -> list[ScenarioSpec]:
    city_ids = list(args.cities) if args.cities else dataset.load_split(args.split)
    scenario_specs: list[ScenarioSpec] = []
    for city_id in city_ids:
        available_scenarios = dataset.scenarios_for_city(city_id)
        requested = list(args.scenarios) if args.scenarios else available_scenarios
        for scenario_name in requested:
            scenario_specs.append(dataset.build_scenario_spec(city_id, scenario_name))
    return scenario_specs


def default_env_config() -> EnvConfig:
    return EnvConfig(
        simulator_interval=1,
        decision_interval=5,
        min_green_time=10,
        thread_num=1,
        max_episode_seconds=300,
        observation=ObservationConfig(),
        reward=RewardConfig(variant="wait_queue_throughput"),
    )


def token_length(inference: DistrictLLMInference, text: str) -> int | None:
    tokenizer = inference.tokenizer
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text)
    return int(len(encoded["input_ids"]))


def prompt_near_model_limit(inference: DistrictLLMInference, prompt_token_length: int | None) -> bool | None:
    if prompt_token_length is None or inference.tokenizer is None:
        return None
    model_max_length = getattr(inference.tokenizer, "model_max_length", None)
    if model_max_length is None or model_max_length <= 0 or model_max_length > 1_000_000:
        return None
    return bool(prompt_token_length >= int(0.9 * model_max_length))


def average(values: list[int | float | None]) -> float:
    filtered = [float(value) for value in values if value is not None]
    return float(mean(filtered)) if filtered else 0.0


def median_or_zero(values: list[int | float | None]) -> float:
    filtered = [float(value) for value in values if value is not None]
    return float(median(filtered)) if filtered else 0.0


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
