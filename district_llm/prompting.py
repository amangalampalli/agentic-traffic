from __future__ import annotations

from district_llm.schema import DISTRICT_STRATEGIES, PHASE_BIASES, PRIORITY_CORRIDORS, DistrictAction, DistrictStateSummary


def format_district_prompt(summary: DistrictStateSummary) -> str:
    return "\n".join(
        [
            "### DISTRICT ACTION SCHEMA",
            f"strategy: {'|'.join(DISTRICT_STRATEGIES)}",
            f"phase_bias: {'|'.join(PHASE_BIASES)}",
            f"priority_corridor: {'|'.join(PRIORITY_CORRIDORS)}|none",
            "duration_steps: integer 1..20",
            "target_intersections: up to 8 ids",
            "",
            "### DISTRICT STATE",
            summary.to_prompt_text(),
            "",
            "### DECISION",
        ]
    )


def format_sft_text(summary: DistrictStateSummary, action: DistrictAction) -> str:
    return f"{format_district_prompt(summary)}\n{action.to_pretty_json()}"
