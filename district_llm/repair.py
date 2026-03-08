from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from district_llm.schema import (
    DISTRICT_STRATEGIES,
    PHASE_BIASES,
    PRIORITY_CORRIDORS,
    CandidateIntersection,
    DistrictAction,
    DistrictStateSummary,
    candidate_priority_score,
    canonicalize_target_intersections,
)


INTERSECTION_ID_PATTERN = re.compile(r"\bi_\d+\b")


@dataclass(frozen=True)
class RepairConfig:
    allow_only_visible_candidates: bool = True
    max_target_intersections: int = 3
    fallback_on_empty_targets: bool = True
    fallback_mode: str = "heuristic"


@dataclass
class RepairReport:
    raw_targets: list[str] = field(default_factory=list)
    repaired_targets: list[str] = field(default_factory=list)
    invalid_ids_removed: list[str] = field(default_factory=list)
    non_visible_ids_removed: list[str] = field(default_factory=list)
    deduplicated: bool = False
    truncated: bool = False
    fallback_used: bool = False
    fallback_mode: str | None = None
    empty_after_filtering: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_targets": list(self.raw_targets),
            "repaired_targets": list(self.repaired_targets),
            "invalid_ids_removed": list(self.invalid_ids_removed),
            "non_visible_ids_removed": list(self.non_visible_ids_removed),
            "deduplicated": bool(self.deduplicated),
            "truncated": bool(self.truncated),
            "fallback_used": bool(self.fallback_used),
            "fallback_mode": self.fallback_mode,
            "empty_after_filtering": bool(self.empty_after_filtering),
        }


def normalize_candidate_intersections(
    payload: list[CandidateIntersection | dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in payload or []:
        if isinstance(item, CandidateIntersection):
            normalized.append(item.to_dict())
        elif isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def parse_candidate_intersections_from_text(text: str) -> list[dict[str, Any]]:
    if "candidate_intersections:" not in text:
        return []

    candidates: list[dict[str, Any]] = []
    capture = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "candidate_intersections:":
            capture = True
            continue
        if not capture:
            continue
        if stripped == "- none":
            continue
        if not stripped.startswith("- "):
            if stripped.endswith(":"):
                break
            continue
        fields = stripped[2:].split()
        if not fields:
            continue
        candidate: dict[str, Any] = {
            "intersection_id": fields[0],
            "queue_total": 0.0,
            "wait_total": 0.0,
            "outgoing_load": 0.0,
            "current_phase": 0,
            "is_boundary": False,
            "spillback_risk": False,
            "incident_proximity": False,
            "overload_marker": False,
            "event_proximity": False,
            "corridor_alignment": "BALANCED",
            "selection_reasons": [],
        }
        for token in fields[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "q":
                candidate["queue_total"] = float(value)
            elif key == "w":
                candidate["wait_total"] = float(value)
            elif key == "out":
                candidate["outgoing_load"] = float(value)
            elif key == "phase":
                candidate["current_phase"] = int(value)
            elif key == "boundary":
                candidate["is_boundary"] = value == "1"
            elif key == "spillback":
                candidate["spillback_risk"] = value == "1"
            elif key == "incident":
                candidate["incident_proximity"] = value == "1"
            elif key == "overload":
                candidate["overload_marker"] = value == "1"
            elif key == "event":
                candidate["event_proximity"] = value == "1"
            elif key == "align":
                candidate["corridor_alignment"] = value
            elif key == "reasons":
                candidate["selection_reasons"] = [] if value == "none" else value.split("|")
        candidates.append(candidate)
    return normalized_candidate_intersections_from_dicts(candidates)


def normalized_candidate_intersections_from_dicts(
    payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in payload:
        try:
            normalized.append(
                CandidateIntersection(
                    intersection_id=str(item.get("intersection_id", "")).strip(),
                    queue_total=float(item.get("queue_total", 0.0)),
                    wait_total=float(item.get("wait_total", 0.0)),
                    outgoing_load=float(item.get("outgoing_load", 0.0)),
                    current_phase=int(item.get("current_phase", 0)),
                    is_boundary=bool(item.get("is_boundary", False)),
                    spillback_risk=bool(item.get("spillback_risk", False)),
                    incident_proximity=bool(item.get("incident_proximity", False)),
                    overload_marker=bool(item.get("overload_marker", False)),
                    event_proximity=bool(item.get("event_proximity", False)),
                    corridor_alignment=str(item.get("corridor_alignment", "BALANCED")),
                    selection_reasons=list(item.get("selection_reasons", [])),
                ).to_dict()
            )
        except Exception:
            continue
    return normalized


def candidate_intersections_from_context(
    summary: DistrictStateSummary | dict[str, Any] | None = None,
    prompt_text: str | None = None,
) -> list[dict[str, Any]]:
    if isinstance(summary, DistrictStateSummary):
        return normalize_candidate_intersections(summary.candidate_intersections)
    if isinstance(summary, dict):
        if "candidate_intersections" in summary:
            return normalize_candidate_intersections(summary.get("candidate_intersections"))
        state_payload = summary.get("state")
        if isinstance(state_payload, dict) and "candidate_intersections" in state_payload:
            return normalize_candidate_intersections(state_payload.get("candidate_intersections"))
    if prompt_text:
        return parse_candidate_intersections_from_text(prompt_text)
    return []


def fallback_target_intersections(
    summary: DistrictStateSummary | dict[str, Any] | None = None,
    prompt_text: str | None = None,
    max_target_intersections: int = 3,
    strategy: str | None = None,
    priority_corridor: str | None = None,
    phase_bias: str | None = None,
    focus_scores: dict[str, float] | None = None,
) -> list[str]:
    candidate_intersections = candidate_intersections_from_context(summary=summary, prompt_text=prompt_text)
    if candidate_intersections:
        ordered_candidates = sorted(
            candidate_intersections,
            key=lambda item: (
                -(
                    candidate_priority_score(item)
                    + _focus_score_bonus(item, focus_scores)
                    + _strategy_target_bonus(
                        candidate=item,
                        strategy=strategy,
                        priority_corridor=priority_corridor,
                        phase_bias=phase_bias,
                    )
                ),
                -float(item.get("queue_total", 0.0)),
                -float(item.get("wait_total", 0.0)),
                -float(item.get("outgoing_load", 0.0)),
                str(item.get("intersection_id", "")),
            ),
        )
        ordered_ids = canonicalize_target_intersections(
            [item["intersection_id"] for item in ordered_candidates],
            ordered_candidates,
            limit=max_target_intersections,
        )
        return ordered_ids[:max_target_intersections]

    if isinstance(summary, DistrictStateSummary):
        return [item.intersection_id for item in summary.top_congested_intersections[:max_target_intersections]]
    if isinstance(summary, dict):
        top_congested = summary.get("top_congested_intersections") or summary.get("state", {}).get("top_congested_intersections", [])
        return [
            str(item.get("intersection_id"))
            for item in top_congested[:max_target_intersections]
            if str(item.get("intersection_id", "")).strip()
        ]
    if prompt_text:
        return list(dict.fromkeys(INTERSECTION_ID_PATTERN.findall(prompt_text)))[:max_target_intersections]
    return []


def _focus_score_bonus(candidate: dict[str, Any], focus_scores: dict[str, float] | None) -> float:
    if not focus_scores:
        return 0.0
    max_focus = max(max(focus_scores.values()), 1.0)
    return 4.0 * float(focus_scores.get(str(candidate.get("intersection_id", "")), 0.0)) / max_focus


def _strategy_target_bonus(
    candidate: dict[str, Any],
    strategy: str | None,
    priority_corridor: str | None,
    phase_bias: str | None,
) -> float:
    reasons = set(candidate.get("selection_reasons", []))
    corridor_alignment = str(candidate.get("corridor_alignment", "BALANCED"))
    bonus = 0.0

    if strategy == "incident_response":
        bonus += 4.0 * float(bool(candidate.get("incident_proximity", False)))
    elif strategy == "clear_spillback":
        bonus += 4.0 * float(bool(candidate.get("spillback_risk", False)))
        bonus += 1.0 * float(bool(candidate.get("is_boundary", False)))
    elif strategy == "drain_inbound":
        bonus += 4.0 * float(bool(candidate.get("is_boundary", False)))
        bonus += 1.0 * float(bool(candidate.get("spillback_risk", False)))
    elif strategy == "drain_outbound":
        bonus += 4.0 * float("outgoing" in reasons)
        bonus += 1.0 * float(bool(candidate.get("spillback_risk", False)))
    elif strategy == "arterial_priority":
        bonus += 2.0 * float(bool(candidate.get("is_boundary", False)))
        bonus += 1.5 * float(bool(candidate.get("overload_marker", False)))
        bonus += 1.5 * float(bool(candidate.get("event_proximity", False)))
    elif strategy == "favor_NS":
        bonus += 4.0 * float(corridor_alignment == "NS")
    elif strategy == "favor_EW":
        bonus += 4.0 * float(corridor_alignment == "EW")

    if priority_corridor in {"NS", "EW"}:
        bonus += 1.5 * float(corridor_alignment == priority_corridor)
    elif priority_corridor == "inbound":
        bonus += 1.5 * float(bool(candidate.get("is_boundary", False)))
    elif priority_corridor == "outbound":
        bonus += 1.5 * float("outgoing" in reasons)
    elif priority_corridor == "arterial":
        bonus += 0.75 * float(bool(candidate.get("is_boundary", False)))

    if phase_bias in {"NS", "EW"}:
        bonus += 0.5 * float(corridor_alignment == phase_bias)

    return bonus


def extract_visible_candidate_ids(
    summary: DistrictStateSummary | dict[str, Any] | None = None,
    prompt_text: str | None = None,
) -> list[str]:
    candidate_intersections = candidate_intersections_from_context(summary=summary, prompt_text=prompt_text)
    if candidate_intersections:
        return [item["intersection_id"] for item in candidate_intersections]
    if prompt_text:
        return list(dict.fromkeys(INTERSECTION_ID_PATTERN.findall(prompt_text)))
    return []


def sanitize_action_payload(
    payload: dict[str, Any] | None,
    summary: DistrictStateSummary | dict[str, Any] | None = None,
    prompt_text: str | None = None,
    config: RepairConfig | None = None,
) -> tuple[DistrictAction, RepairReport]:
    config = config or RepairConfig()
    payload = dict(payload or {})
    candidate_intersections = candidate_intersections_from_context(summary=summary, prompt_text=prompt_text)
    visible_candidate_ids = [item["intersection_id"] for item in candidate_intersections]
    visible_candidate_set = set(visible_candidate_ids)

    raw_target_payload = payload.get("target_intersections", [])
    if isinstance(raw_target_payload, str):
        raw_targets = INTERSECTION_ID_PATTERN.findall(raw_target_payload)
    elif isinstance(raw_target_payload, (list, tuple)):
        raw_targets = [str(item).strip() for item in raw_target_payload if str(item).strip()]
    else:
        raw_targets = []

    report = RepairReport(raw_targets=list(raw_targets))
    deduped_targets: list[str] = []
    seen: set[str] = set()
    for item in raw_targets:
        if item in seen:
            report.deduplicated = True
            continue
        seen.add(item)
        deduped_targets.append(item)

    filtered_targets: list[str] = []
    for item in deduped_targets:
        if not INTERSECTION_ID_PATTERN.fullmatch(item):
            report.invalid_ids_removed.append(item)
            continue
        if config.allow_only_visible_candidates and visible_candidate_set and item not in visible_candidate_set:
            report.non_visible_ids_removed.append(item)
            continue
        filtered_targets.append(item)

    if len(filtered_targets) > int(config.max_target_intersections):
        report.truncated = True
    filtered_targets = canonicalize_target_intersections(
        filtered_targets,
        candidate_intersections,
        limit=int(config.max_target_intersections),
    )

    if not filtered_targets:
        report.empty_after_filtering = bool(raw_targets)
        if config.fallback_on_empty_targets:
            report.fallback_used = True
            report.fallback_mode = config.fallback_mode
            if config.fallback_mode == "heuristic":
                filtered_targets = fallback_target_intersections(
                    summary=summary,
                    prompt_text=prompt_text,
                    max_target_intersections=int(config.max_target_intersections),
                )
            elif config.fallback_mode == "hold":
                filtered_targets = []
            elif config.fallback_mode == "none":
                filtered_targets = []
            else:
                raise ValueError(f"Unsupported fallback_mode '{config.fallback_mode}'.")

    strategy = str(payload.get("strategy", "hold"))
    if strategy not in DISTRICT_STRATEGIES:
        strategy = "hold"

    priority_corridor = payload.get("priority_corridor")
    if priority_corridor is not None:
        priority_corridor = str(priority_corridor)
    if priority_corridor not in PRIORITY_CORRIDORS:
        priority_corridor = None

    phase_bias = str(payload.get("phase_bias", "NONE"))
    if phase_bias not in PHASE_BIASES:
        phase_bias = "NONE"

    duration_steps_raw = payload.get("duration_steps", 1)
    try:
        duration_steps = int(duration_steps_raw)
    except (TypeError, ValueError):
        duration_steps = 1
    duration_steps = max(1, min(duration_steps, 20))

    if config.fallback_mode == "hold" and report.fallback_used and not filtered_targets:
        action = DistrictAction.default_hold(duration_steps=duration_steps)
    else:
        action = DistrictAction(
            strategy=strategy,
            priority_corridor=priority_corridor,
            target_intersections=filtered_targets,
            phase_bias=phase_bias,
            duration_steps=duration_steps,
        ).validate()

    report.repaired_targets = list(action.target_intersections)
    return action, report
