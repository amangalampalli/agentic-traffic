from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from district_llm.metrics import aggregate_target_metrics, compute_target_metrics, safe_ratio, target_failure_buckets
from district_llm.repair import RepairConfig, extract_visible_candidate_ids, sanitize_action_payload
from district_llm.schema import DistrictAction
from env.utils import build_topology

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline evaluation for district-LLM outputs."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--debug-examples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--restrict-targets-to-visible-summary", action="store_true")
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
        "--report-before-after-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_rows(path: str | Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def extract_json_object(payload: str) -> str:
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return payload[start : end + 1]


def load_model_and_tokenizer(model_path: str, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if (model_dir / "adapter_config.json").exists():
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Evaluating a LoRA adapter requires the 'peft' package."
            ) from exc
        model = AutoPeftModelForCausalLM.from_pretrained(model_path)
    else:
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_path).to(target_device)
    model.eval()
    return model, tokenizer


def build_generation_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def generate_response(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    import torch

    prompt = build_generation_prompt(tokenizer, messages)
    device = getattr(model, "device", None)
    inputs = tokenizer(prompt, return_tensors="pt")
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def parse_prediction(payload: str) -> tuple[bool, bool, dict[str, Any] | None]:
    try:
        json_payload = json.loads(extract_json_object(payload))
    except Exception:
        return False, False, None
    try:
        action = DistrictAction.from_dict(json_payload)
    except Exception:
        return True, False, json_payload
    return True, True, action.to_dict()


class DistrictTopologyIndex:
    def __init__(self, generated_root: str | Path):
        self.generated_root = Path(generated_root)
        self._cache: dict[str, dict[str, set[str]]] = {}

    def district_intersections(self, city_id: str, district_id: str) -> set[str]:
        if city_id not in self._cache:
            roadnet_path = self.generated_root / city_id / "roadnet.json"
            district_map_path = self.generated_root / city_id / "district_map.json"
            metadata_path = self.generated_root / city_id / "metadata.json"
            _, districts = build_topology(
                roadnet_path=roadnet_path,
                district_map_path=district_map_path,
                metadata_path=metadata_path,
            )
            self._cache[city_id] = {
                key: set(value.intersection_ids)
                for key, value in districts.items()
            }
        return self._cache[city_id].get(district_id, set())


def field_accuracy(pred: dict[str, Any] | None, gt: dict[str, Any], field: str) -> float:
    if pred is None:
        return 0.0
    return float(pred.get(field) == gt.get(field))


def invalid_target_fraction(pred_targets: list[str], district_candidates: set[str]) -> float:
    if not pred_targets:
        return 0.0
    invalid_count = sum(1 for item in pred_targets if item not in district_candidates)
    return safe_ratio(invalid_count, len(pred_targets))


def evaluate_rows(
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    max_new_tokens: int,
    topology_index: DistrictTopologyIndex,
    restrict_targets_to_visible_summary: bool,
    debug_examples: int,
    repair_config: RepairConfig,
    report_before_after_repair: bool,
) -> dict[str, Any]:
    json_valid_count = 0
    schema_valid_count = 0
    field_totals_before = Counter()
    field_totals_after = Counter()
    full_object_correct_before = 0
    full_object_correct_after = 0
    target_rows_before: list[dict[str, float]] = []
    target_rows_after: list[dict[str, float]] = []
    restricted_target_rows_before: list[dict[str, float]] = []
    restricted_target_rows_after: list[dict[str, float]] = []
    invalid_rates_before: list[float] = []
    invalid_rates_after: list[float] = []
    fallback_used_count = 0
    failure_buckets = Counter()
    debug_rows = []

    progress = (
        tqdm(total=len(rows), desc="eval", dynamic_ncols=True)
        if tqdm is not None
        else None
    )

    try:
        for row in rows:
            messages = row["messages"]
            ground_truth = json.loads(messages[2]["content"])
            raw_prediction = generate_response(
                model=model,
                tokenizer=tokenizer,
                messages=messages[:2],
                max_new_tokens=max_new_tokens,
            )
            json_valid, schema_valid, prediction_before = parse_prediction(raw_prediction)
            repaired_action, repair_report = sanitize_action_payload(
                payload=prediction_before if json_valid else None,
                summary=row,
                prompt_text=messages[1]["content"],
                config=repair_config,
            )
            prediction_after = repaired_action.to_dict()
            json_valid_count += int(json_valid)
            schema_valid_count += int(schema_valid)
            fallback_used_count += int(repair_report.fallback_used)

            field_totals_before["strategy"] += field_accuracy(prediction_before, ground_truth, "strategy")
            field_totals_before["priority_corridor"] += field_accuracy(prediction_before, ground_truth, "priority_corridor")
            field_totals_before["phase_bias"] += field_accuracy(prediction_before, ground_truth, "phase_bias")
            field_totals_before["duration_steps"] += field_accuracy(prediction_before, ground_truth, "duration_steps")

            field_totals_after["strategy"] += field_accuracy(prediction_after, ground_truth, "strategy")
            field_totals_after["priority_corridor"] += field_accuracy(prediction_after, ground_truth, "priority_corridor")
            field_totals_after["phase_bias"] += field_accuracy(prediction_after, ground_truth, "phase_bias")
            field_totals_after["duration_steps"] += field_accuracy(prediction_after, ground_truth, "duration_steps")

            if prediction_before == ground_truth:
                full_object_correct_before += 1
            if prediction_after == ground_truth:
                full_object_correct_after += 1

            pred_targets_before = [] if prediction_before is None else list(prediction_before.get("target_intersections", []))
            pred_targets_after = list(prediction_after.get("target_intersections", []))
            gt_targets = list(ground_truth.get("target_intersections", []))
            visible_candidates = set(
                extract_visible_candidate_ids(summary=row, prompt_text=messages[1]["content"])
            )
            district_candidates = topology_index.district_intersections(
                city_id=row["city_id"],
                district_id=row["district_id"],
            )
            invalid_before = [item for item in pred_targets_before if item not in district_candidates]
            invalid_after = [item for item in pred_targets_after if item not in district_candidates]
            non_visible_before = [
                item for item in pred_targets_before
                if visible_candidates and item not in visible_candidates
            ]

            metrics_before = compute_target_metrics(pred_targets_before, gt_targets)
            metrics_after = compute_target_metrics(pred_targets_after, gt_targets)
            target_rows_before.append(metrics_before)
            target_rows_after.append(metrics_after)
            invalid_rates_before.append(invalid_target_fraction(pred_targets_before, district_candidates))
            invalid_rates_after.append(invalid_target_fraction(pred_targets_after, district_candidates))

            if restrict_targets_to_visible_summary:
                filtered_pred_before = [item for item in pred_targets_before if item in visible_candidates]
                filtered_pred_after = [item for item in pred_targets_after if item in visible_candidates]
                filtered_gt = [item for item in gt_targets if item in visible_candidates]
                restricted_target_rows_before.append(
                    compute_target_metrics(filtered_pred_before, filtered_gt)
                )
                restricted_target_rows_after.append(
                    compute_target_metrics(filtered_pred_after, filtered_gt)
                )

            for failure_bucket in set(
                target_failure_buckets(
                    pred_list=pred_targets_before,
                    gt_list=gt_targets,
                    visible_candidates=visible_candidates,
                    invalid_ids=invalid_before,
                    non_visible_ids=non_visible_before,
                    repaired_targets=pred_targets_after,
                    fallback_used=repair_report.fallback_used,
                )
            ):
                failure_buckets[failure_bucket] += 1

            if len(debug_rows) < debug_examples:
                debug_rows.append(
                    {
                        "district_summary": messages[1]["content"],
                        "predicted_json_raw": raw_prediction,
                        "predicted_json_parsed_before_repair": prediction_before,
                        "predicted_json_parsed_after_repair": prediction_after,
                        "ground_truth_json": ground_truth,
                        "target_intersections_metrics_before_repair": metrics_before,
                        "target_intersections_metrics_after_repair": metrics_after,
                        "repair_report": repair_report.to_dict(),
                        "visible_candidate_ids": sorted(visible_candidates),
                        "failure_buckets": sorted(
                            set(
                                target_failure_buckets(
                                    pred_list=pred_targets_before,
                                    gt_list=gt_targets,
                                    visible_candidates=visible_candidates,
                                    invalid_ids=invalid_before,
                                    non_visible_ids=non_visible_before,
                                    repaired_targets=pred_targets_after,
                                    fallback_used=repair_report.fallback_used,
                                )
                            )
                        ),
                    }
                )
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    total_rows = max(1, len(rows))
    results = {
        "num_examples": len(rows),
        "json_validity_rate": float(json_valid_count) / total_rows,
        "schema_validity_rate": float(schema_valid_count) / total_rows,
        "field_accuracy": {
            "strategy": float(field_totals_before["strategy"]) / total_rows,
            "priority_corridor": float(field_totals_before["priority_corridor"]) / total_rows,
            "phase_bias": float(field_totals_before["phase_bias"]) / total_rows,
            "duration_steps": float(field_totals_before["duration_steps"]) / total_rows,
        },
        "field_accuracy_after_repair": {
            "strategy": float(field_totals_after["strategy"]) / total_rows,
            "priority_corridor": float(field_totals_after["priority_corridor"]) / total_rows,
            "phase_bias": float(field_totals_after["phase_bias"]) / total_rows,
            "duration_steps": float(field_totals_after["duration_steps"]) / total_rows,
        },
        "target_intersections_before_repair": aggregate_target_metrics(target_rows_before),
        "target_intersections_after_repair": aggregate_target_metrics(target_rows_after),
        "target_intersections": aggregate_target_metrics(target_rows_after),
        "target_intersections_failure_buckets": dict(sorted(failure_buckets.items())),
        "exact_full_object_accuracy": float(full_object_correct_before) / total_rows,
        "exact_full_object_accuracy_after_repair": float(full_object_correct_after) / total_rows,
        "debug_examples": debug_rows,
    }
    if restrict_targets_to_visible_summary:
        results["target_intersections_restricted_to_visible_summary_before_repair"] = aggregate_target_metrics(
            restricted_target_rows_before
        )
        results["target_intersections_restricted_to_visible_summary_after_repair"] = aggregate_target_metrics(
            restricted_target_rows_after
        )
        results["target_intersections_restricted_to_visible_summary"] = aggregate_target_metrics(
            restricted_target_rows_after
        )
    if report_before_after_repair:
        results["target_intersections_before_after_repair"] = {
            "invalid_id_rate_before_repair": float(sum(invalid_rates_before) / total_rows),
            "invalid_id_rate_after_repair": float(sum(invalid_rates_after) / total_rows),
            "exact_set_match_before_repair": aggregate_target_metrics(target_rows_before).get("exact_set_match", 0.0),
            "exact_set_match_after_repair": aggregate_target_metrics(target_rows_after).get("exact_set_match", 0.0),
            "jaccard_before_repair": aggregate_target_metrics(target_rows_before).get("jaccard", 0.0),
            "jaccard_after_repair": aggregate_target_metrics(target_rows_after).get("jaccard", 0.0),
            "fallback_used_rate": float(fallback_used_count) / total_rows,
        }
    return results


def print_debug_examples(debug_rows: list[dict[str, Any]]) -> None:
    for index, item in enumerate(debug_rows, start=1):
        print(f"[debug {index}] district_summary:")
        print(item["district_summary"])
        print(f"[debug {index}] predicted_json_raw={item['predicted_json_raw']}")
        print(
            f"[debug {index}] predicted_json_parsed_before_repair="
            f"{json.dumps(item['predicted_json_parsed_before_repair'], sort_keys=True)}"
        )
        print(
            f"[debug {index}] predicted_json_parsed_after_repair="
            f"{json.dumps(item['predicted_json_parsed_after_repair'], sort_keys=True)}"
        )
        print(
            f"[debug {index}] ground_truth_json="
            f"{json.dumps(item['ground_truth_json'], sort_keys=True)}"
        )
        print(
            f"[debug {index}] target_intersections_metrics_before_repair="
            f"{json.dumps(item['target_intersections_metrics_before_repair'], sort_keys=True)}"
        )
        print(
            f"[debug {index}] target_intersections_metrics_after_repair="
            f"{json.dumps(item['target_intersections_metrics_after_repair'], sort_keys=True)}"
        )
        print(
            f"[debug {index}] repair_report="
            f"{json.dumps(item['repair_report'], sort_keys=True)}"
        )
        print(
            f"[debug {index}] visible_candidate_ids="
            f"{json.dumps(item['visible_candidate_ids'], sort_keys=True)}"
        )
        print(f"[debug {index}] failure_buckets={json.dumps(item['failure_buckets'])}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.val_jsonl, max_examples=args.max_examples)
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    topology_index = DistrictTopologyIndex(args.generated_root)
    results = evaluate_rows(
        rows=rows,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        topology_index=topology_index,
        restrict_targets_to_visible_summary=args.restrict_targets_to_visible_summary,
        debug_examples=args.debug_examples,
        repair_config=RepairConfig(
            allow_only_visible_candidates=args.allow_only_visible_candidates,
            max_target_intersections=args.max_target_intersections,
            fallback_on_empty_targets=args.fallback_on_empty_targets,
            fallback_mode=args.fallback_mode,
        ),
        report_before_after_repair=args.report_before_after_repair,
    )
    print(json.dumps({k: v for k, v in results.items() if k != "debug_examples"}, indent=2, sort_keys=True))
    print_debug_examples(results["debug_examples"])


if __name__ == "__main__":
    main()
