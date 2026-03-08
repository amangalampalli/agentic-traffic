from __future__ import annotations

from typing import Any


def safe_ratio(numerator: int | float, denominator: int | float, default_if_empty: float = 0.0) -> float:
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


def target_failure_buckets(
    pred_list: list[str],
    gt_list: list[str],
    visible_candidates: set[str],
    invalid_ids: list[str] | None = None,
    non_visible_ids: list[str] | None = None,
    repaired_targets: list[str] | None = None,
    fallback_used: bool = False,
) -> list[str]:
    buckets: list[str] = []
    pred_set = set(pred_list)
    gt_set = set(gt_list)

    if not pred_list:
        buckets.append("prediction_empty")
    if not gt_list:
        buckets.append("ground_truth_empty")
    if pred_list and gt_list and pred_set == gt_set and pred_list != gt_list:
        buckets.append("same_set_different_order")
    elif pred_set & gt_set:
        buckets.append("partial_overlap")
    elif pred_list and gt_list:
        buckets.append("no_overlap")

    if invalid_ids:
        buckets.append("prediction_contains_invalid_ids")
    if non_visible_ids:
        buckets.append("prediction_contains_ids_not_visible_in_summary")
    if pred_list and visible_candidates and any(item not in visible_candidates for item in pred_list):
        buckets.append("prediction_contains_ids_not_visible_in_summary")
    if fallback_used:
        buckets.append("fallback_used")

    if repaired_targets is not None:
        repaired_set = set(repaired_targets)
        if repaired_set == gt_set and pred_set != gt_set:
            buckets.append("repaired_successfully")
        elif (invalid_ids or non_visible_ids or fallback_used) and repaired_set != gt_set:
            buckets.append("repair_failed")

    return buckets


def average_item_rate(values: list[list[Any]]) -> float:
    numerators = sum(len(item) for item in values)
    denominators = sum(max(len(item), 1) for item in values)
    return safe_ratio(numerators, denominators)
