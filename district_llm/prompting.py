from __future__ import annotations

from district_llm.schema import DISTRICT_STRATEGIES, PHASE_BIASES, PRIORITY_CORRIDORS, DistrictAction, DistrictStateSummary


DEFAULT_MAX_TARGET_INTERSECTIONS = 3


def build_system_prompt(
    max_target_intersections: int = DEFAULT_MAX_TARGET_INTERSECTIONS,
    allow_only_visible_candidates: bool = True,
) -> str:
    candidate_rule = (
        " If candidate_intersections is present, target_intersections must use only ids from that list."
        if allow_only_visible_candidates
        else ""
    )
    return (
        "You are a district traffic coordinator for RL traffic lights. "
        "Return only valid JSON with exactly these keys: "
        "strategy, priority_corridor, target_intersections, phase_bias, duration_steps. "
        f"target_intersections must be a JSON array with at most {int(max_target_intersections)} unique ids."
        f"{candidate_rule} "
        "Do not invent intersection ids. Deduplicate ids. If uncertain, prefer the most congested valid candidates."
    )


def format_district_prompt(
    summary: DistrictStateSummary,
    max_target_intersections: int = DEFAULT_MAX_TARGET_INTERSECTIONS,
    allow_only_visible_candidates: bool = True,
) -> str:
    target_rule = (
        f"target_intersections: up to {int(max_target_intersections)} ids from candidate_intersections only"
        if allow_only_visible_candidates
        else f"target_intersections: up to {int(max_target_intersections)} valid ids"
    )
    return "\n".join(
        [
            "### DISTRICT ACTION SCHEMA",
            f"strategy: {'|'.join(DISTRICT_STRATEGIES)}",
            f"phase_bias: {'|'.join(PHASE_BIASES)}",
            f"priority_corridor: {'|'.join(PRIORITY_CORRIDORS)}|none",
            "duration_steps: integer 1..20",
            target_rule,
            "rules: return only valid JSON; do not invent ids; deduplicate target_intersections",
            "fallback: if uncertain, prefer the most congested visible candidates",
            "",
            "### DISTRICT STATE",
            summary.to_prompt_text(),
            "",
            "### DECISION",
        ]
    )


def format_sft_text(
    summary: DistrictStateSummary,
    action: DistrictAction,
    max_target_intersections: int = DEFAULT_MAX_TARGET_INTERSECTIONS,
    allow_only_visible_candidates: bool = True,
) -> str:
    return (
        f"{format_district_prompt(summary, max_target_intersections=max_target_intersections, allow_only_visible_candidates=allow_only_visible_candidates)}\n"
        f"{action.to_pretty_json()}"
    )
