from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from district_llm.schema import DistrictAction
from env.utils import build_topology

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


INTERSECTION_ID_PATTERN = re.compile(r"\bi_\d+\b")


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
    if hasattr(tokenizer, "apply_chat_template"):
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


def safe_ratio(numerator: int, denominator: int, default_if_empty: float = 0.0) -> float:
    if denominator <= 0:
        return default_if_empty
    return float(numerator) / float(denominator)


def compute_target_metrics(pred_list: list[str], gt_list: list[str]) -> dict[str, float]:
    pred = list(pred_list)
    gt = list(gt_list)
    pred_set = set(pred)
    gt_set = set(gt)
    overlap = pred_set & gt_set
    union = pred_set | gt_set

    both_empty = not pred_set and not gt_set
    precision_default = 1.0 if both_empty else 0.0
    recall_default = 1.0 if both_empty else 0.0
    jaccard_default = 1.0 if both_empty else 0.0
    overlap_rate_default = 1.0 if both_empty else 0.0

    overlap_count = len(overlap)
    return {
        "exact_list_match": float(pred == gt),
        "exact_set_match": float(pred_set == gt_set),
        "overlap_count": float(overlap_count),
        "overlap_rate": safe_ratio(overlap_count, len(gt_set), overlap_rate_default),
        "precision": safe_ratio(overlap_count, len(pred_set), precision_default),
        "recall": safe_ratio(overlap_count, len(gt_set), recall_default),
        "jaccard": safe_ratio(overlap_count, len(union), jaccard_default),
        "hit_at_1": float(overlap_count >= 1),
        "hit_at_2": float(overlap_count >= 2),
        "hit_at_3": float(overlap_count >= 3),
    }


def aggregate_target_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = metric_rows[0].keys()
    return {
        key: float(sum(row[key] for row in metric_rows) / len(metric_rows))
        for key in keys
    }


def extract_visible_candidates(user_content: str) -> set[str]:
    return set(INTERSECTION_ID_PATTERN.findall(user_content))


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


def categorize_target_failure(
    pred_list: list[str],
    gt_list: list[str],
    visible_candidates: set[str],
    district_candidates: set[str],
) -> str | None:
    if pred_list == gt_list:
        return None
    pred_set = set(pred_list)
    gt_set = set(gt_list)

    if not pred_list:
        return "prediction_empty"
    if not gt_list:
        return "ground_truth_empty"

    valid_candidates = set(visible_candidates) | set(district_candidates)
    invalid_ids = [item for item in pred_list if item not in valid_candidates]
    if invalid_ids:
        return "prediction_contains_invalid_intersection_ids"
    if pred_set == gt_set:
        return "same_set_different_order"
    if pred_set & gt_set:
        return "partial_overlap"
    return "no_overlap"


def field_accuracy(pred: dict[str, Any] | None, gt: dict[str, Any], field: str) -> float:
    if pred is None:
        return 0.0
    return float(pred.get(field) == gt.get(field))


def evaluate_rows(
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    max_new_tokens: int,
    topology_index: DistrictTopologyIndex,
    restrict_targets_to_visible_summary: bool,
    debug_examples: int,
) -> dict[str, Any]:
    json_valid_count = 0
    schema_valid_count = 0
    field_totals = Counter()
    full_object_correct = 0
    unrestricted_target_rows: list[dict[str, float]] = []
    restricted_target_rows: list[dict[str, float]] = []
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
            json_valid, schema_valid, prediction = parse_prediction(raw_prediction)
            json_valid_count += int(json_valid)
            schema_valid_count += int(schema_valid)

            field_totals["strategy"] += field_accuracy(prediction, ground_truth, "strategy")
            field_totals["priority_corridor"] += field_accuracy(prediction, ground_truth, "priority_corridor")
            field_totals["phase_bias"] += field_accuracy(prediction, ground_truth, "phase_bias")
            field_totals["duration_steps"] += field_accuracy(prediction, ground_truth, "duration_steps")

            if prediction == ground_truth:
                full_object_correct += 1

            pred_targets = [] if prediction is None else list(prediction.get("target_intersections", []))
            gt_targets = list(ground_truth.get("target_intersections", []))
            visible_candidates = extract_visible_candidates(messages[1]["content"])
            district_candidates = topology_index.district_intersections(
                city_id=row["city_id"],
                district_id=row["district_id"],
            )

            unrestricted_metrics = compute_target_metrics(pred_targets, gt_targets)
            unrestricted_target_rows.append(unrestricted_metrics)

            if restrict_targets_to_visible_summary:
                filtered_pred = [item for item in pred_targets if item in visible_candidates]
                filtered_gt = [item for item in gt_targets if item in visible_candidates]
                restricted_target_rows.append(
                    compute_target_metrics(filtered_pred, filtered_gt)
                )

            failure_bucket = categorize_target_failure(
                pred_list=pred_targets,
                gt_list=gt_targets,
                visible_candidates=visible_candidates,
                district_candidates=district_candidates,
            )
            if failure_bucket is not None:
                failure_buckets[failure_bucket] += 1

            if len(debug_rows) < debug_examples:
                debug_rows.append(
                    {
                        "district_summary": messages[1]["content"],
                        "predicted_json_raw": raw_prediction,
                        "predicted_json_parsed": prediction,
                        "ground_truth_json": ground_truth,
                        "target_intersections_metrics": unrestricted_metrics,
                        "failure_bucket": failure_bucket,
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
            "strategy": float(field_totals["strategy"]) / total_rows,
            "priority_corridor": float(field_totals["priority_corridor"]) / total_rows,
            "phase_bias": float(field_totals["phase_bias"]) / total_rows,
            "duration_steps": float(field_totals["duration_steps"]) / total_rows,
        },
        "target_intersections": aggregate_target_metrics(unrestricted_target_rows),
        "target_intersections_failure_buckets": dict(sorted(failure_buckets.items())),
        "exact_full_object_accuracy": float(full_object_correct) / total_rows,
        "debug_examples": debug_rows,
    }
    if restrict_targets_to_visible_summary:
        results["target_intersections_restricted_to_visible_summary"] = aggregate_target_metrics(
            restricted_target_rows
        )
    return results


def print_debug_examples(debug_rows: list[dict[str, Any]]) -> None:
    for index, item in enumerate(debug_rows, start=1):
        print(f"[debug {index}] district_summary:")
        print(item["district_summary"])
        print(f"[debug {index}] predicted_json_raw={item['predicted_json_raw']}")
        print(
            f"[debug {index}] predicted_json_parsed="
            f"{json.dumps(item['predicted_json_parsed'], sort_keys=True)}"
        )
        print(
            f"[debug {index}] ground_truth_json="
            f"{json.dumps(item['ground_truth_json'], sort_keys=True)}"
        )
        print(
            f"[debug {index}] target_intersections_metrics="
            f"{json.dumps(item['target_intersections_metrics'], sort_keys=True)}"
        )
        print(f"[debug {index}] failure_bucket={item['failure_bucket']}")


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
    )
    print(json.dumps({k: v for k, v in results.items() if k != "debug_examples"}, indent=2, sort_keys=True))
    print_debug_examples(results["debug_examples"])


if __name__ == "__main__":
    main()
