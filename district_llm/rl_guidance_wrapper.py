from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
import hashlib
from time import perf_counter
from typing import Any

import numpy as np
import torch

from district_llm.heuristic_guidance import (
    HeuristicGuidanceConfig,
    generate_heuristic_guidance,
)
from district_llm.inference import DistrictLLMInference, DistrictLLMInferenceResult
from district_llm.repair import RepairReport
from district_llm.schema import CandidateIntersection, DistrictAction, DistrictStateSummary
from district_llm.summary_builder import DistrictStateSummaryBuilder
from district_llm.teachers import RLCheckpointTeacher


WRAPPER_MODES: tuple[str, ...] = (
    "no_op",
    "target_only_soft",
    "target_only_medium",
    "corridor_soft",
    "global_soft",
    "current_legacy",
)
FALLBACK_POLICIES: tuple[str, ...] = (
    "no_op",
    "hold_previous",
    "heuristic_weak",
)
GATING_MODES: tuple[str, ...] = (
    "always_on",
    "incident_or_spillback",
    "queue_threshold",
    "imbalance_threshold",
    "queue_or_imbalance",
    "combined",
)
BIAS_DECAY_SCHEDULES: tuple[str, ...] = (
    "linear",
)
STRATEGY_BIAS_MULTIPLIERS: dict[str, float] = {
    "hold": 0.0,
    "favor_NS": 1.0,
    "favor_EW": 1.0,
    "drain_inbound": 1.05,
    "drain_outbound": 1.05,
    "clear_spillback": 1.1,
    "incident_response": 1.15,
    "arterial_priority": 1.05,
}


@dataclass(frozen=True)
class GuidanceInfluenceConfig:
    """
    Conservative inference-time wrapper around the fixed DQN policy.

    The DQN checkpoint remains unchanged. Guidance is treated as a weak prior
    and only biases Q-values slightly before greedy action selection.
    """

    wrapper_mode: str = "target_only_soft"
    bias_strength: float = 0.12
    target_only_bias_strength: float = 0.18
    corridor_bias_strength: float = 0.05
    max_intersections_affected: int = 3
    guidance_refresh_steps: int = 5
    guidance_persistence_steps: int = 3
    max_guidance_duration: int = 6
    apply_global_bias: bool = False
    apply_target_only: bool = True
    gating_mode: str = "always_on"
    min_avg_queue_for_guidance: float = 150.0
    min_queue_imbalance_for_guidance: float = 20.0
    require_incident_or_spillback: bool = False
    allow_guidance_in_normal_conditions: bool = True
    enable_bias_decay: bool = True
    bias_decay_schedule: str = "linear"
    fallback_policy: str = "hold_previous"
    log_guidance_debug: bool = False
    max_debug_chars: int = 240

    def validate(self) -> "GuidanceInfluenceConfig":
        if self.wrapper_mode not in WRAPPER_MODES:
            raise ValueError(
                f"Unsupported wrapper_mode '{self.wrapper_mode}'. Expected one of {WRAPPER_MODES}."
            )
        if self.fallback_policy not in FALLBACK_POLICIES:
            raise ValueError(
                f"Unsupported fallback_policy '{self.fallback_policy}'. Expected one of {FALLBACK_POLICIES}."
            )
        if self.gating_mode not in GATING_MODES:
            raise ValueError(
                f"Unsupported gating_mode '{self.gating_mode}'. Expected one of {GATING_MODES}."
            )
        if self.bias_decay_schedule not in BIAS_DECAY_SCHEDULES:
            raise ValueError(
                f"Unsupported bias_decay_schedule '{self.bias_decay_schedule}'. "
                f"Expected one of {BIAS_DECAY_SCHEDULES}."
            )
        if self.guidance_refresh_steps < 1:
            raise ValueError("guidance_refresh_steps must be at least 1.")
        if self.guidance_persistence_steps < 1:
            raise ValueError("guidance_persistence_steps must be at least 1.")
        if self.max_guidance_duration < 1:
            raise ValueError("max_guidance_duration must be at least 1.")
        if self.max_intersections_affected < 1:
            raise ValueError("max_intersections_affected must be at least 1.")
        return self


@dataclass(frozen=True)
class RLPolicyDecision:
    q_values: np.ndarray
    actions: np.ndarray


@dataclass
class GuidanceDecision:
    source: str
    action: DistrictAction
    runtime_seconds: float
    raw_text: str | None = None
    parsed_payload_before_repair: dict[str, Any] | None = None
    repair_report: RepairReport | None = None
    json_valid: bool = True
    schema_valid_before_repair: bool = True
    provider_error: str | None = None
    fallback_policy_applied: str | None = None

    @property
    def repair_applied(self) -> bool:
        report = self.repair_report
        if report is None:
            return False
        return any(
            (
                report.invalid_ids_removed,
                report.non_visible_ids_removed,
                report.deduplicated,
                report.truncated,
                report.fallback_used,
                report.empty_after_filtering,
            )
        )

    @property
    def invalid_before_repair(self) -> bool:
        report = self.repair_report
        if self.provider_error:
            return True
        if not self.json_valid or not self.schema_valid_before_repair:
            return True
        if report is None:
            return False
        return bool(
            report.invalid_ids_removed
            or report.non_visible_ids_removed
            or report.empty_after_filtering
        )

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "runtime_seconds": float(self.runtime_seconds),
            "action": self.action.to_dict(),
            "raw_text": self.raw_text,
            "parsed_payload_before_repair": self.parsed_payload_before_repair,
            "repair_report": None if self.repair_report is None else self.repair_report.to_dict(),
            "json_valid": bool(self.json_valid),
            "schema_valid_before_repair": bool(self.schema_valid_before_repair),
            "repair_applied": bool(self.repair_applied),
            "invalid_before_repair": bool(self.invalid_before_repair),
            "provider_error": self.provider_error,
            "fallback_policy_applied": self.fallback_policy_applied,
        }


@dataclass(frozen=True)
class GuidanceApplicationPlan:
    wrapper_mode: str
    scope: str
    affected_intersections: tuple[str, ...]
    targeted_intersections: tuple[str, ...]
    target_candidate_ids: tuple[str, ...]
    priority_direction: str | None
    strength_scale: float
    base_bias_strength: float
    target_bias_strength: float
    corridor_bias_strength: float
    apply_global_bias: bool
    apply_target_only: bool
    max_intersections_affected: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "wrapper_mode": self.wrapper_mode,
            "scope": self.scope,
            "affected_intersections": list(self.affected_intersections),
            "targeted_intersections": list(self.targeted_intersections),
            "target_candidate_ids": list(self.target_candidate_ids),
            "priority_direction": self.priority_direction,
            "strength_scale": float(self.strength_scale),
            "base_bias_strength": float(self.base_bias_strength),
            "target_bias_strength": float(self.target_bias_strength),
            "corridor_bias_strength": float(self.corridor_bias_strength),
            "apply_global_bias": bool(self.apply_global_bias),
            "apply_target_only": bool(self.apply_target_only),
            "max_intersections_affected": int(self.max_intersections_affected),
        }


@dataclass
class ActiveDistrictGuidance:
    district_id: str
    summary: DistrictStateSummary
    decision: GuidanceDecision
    application_plan: GuidanceApplicationPlan
    generated_step: int
    expires_step: int
    fallback_used: bool = False


@dataclass(frozen=True)
class GuidanceGateDecision:
    allowed: bool
    gating_mode: str
    triggered_conditions: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    avg_queue: float
    queue_imbalance: float
    incident_flag: bool
    spillback_risk: bool
    overload_flag: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": bool(self.allowed),
            "gating_mode": self.gating_mode,
            "triggered_conditions": list(self.triggered_conditions),
            "blocked_reasons": list(self.blocked_reasons),
            "avg_queue": float(self.avg_queue),
            "queue_imbalance": float(self.queue_imbalance),
            "incident_flag": bool(self.incident_flag),
            "spillback_risk": bool(self.spillback_risk),
            "overload_flag": bool(self.overload_flag),
        }


@dataclass
class GuidanceRefreshTrace:
    mode_source: str
    district_id: str
    decision_step: int
    summary_hash: str
    summary_excerpt: str
    summary_payload: dict[str, Any]
    guidance: dict[str, Any]
    repaired_guidance: dict[str, Any]
    fallback_used: bool
    fallback_policy: str
    application_plan: dict[str, Any]
    applied_biases: dict[str, float]
    gate_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_source": self.mode_source,
            "district_id": self.district_id,
            "decision_step": int(self.decision_step),
            "summary_hash": self.summary_hash,
            "summary_excerpt": self.summary_excerpt,
            "summary": self.summary_payload,
            "raw_guidance": self.guidance,
            "repaired_guidance": self.repaired_guidance,
            "fallback_used": bool(self.fallback_used),
            "fallback_policy": self.fallback_policy,
            "application_plan": self.application_plan,
            "applied_biases": self.applied_biases,
            "gate_decision": self.gate_decision,
        }


@dataclass
class WrapperEpisodeStats:
    step_count: int = 0
    steps_with_active_guidance: int = 0
    guidance_refresh_count: int = 0
    guidance_blocked_step_count: int = 0
    guidance_blocked_refresh_count: int = 0
    bias_application_count: int = 0
    noop_guidance_events: int = 0
    fallback_event_count: int = 0
    total_affected_intersections: int = 0
    total_targeted_intersections: int = 0
    total_bias_magnitude: float = 0.0
    max_bias_magnitude: float = 0.0

    def to_dict(self) -> dict[str, float]:
        refresh_count = max(1, self.guidance_refresh_count)
        return {
            "num_guidance_refreshes": float(self.guidance_refresh_count),
            "num_steps_guidance_blocked_by_gate": float(self.guidance_blocked_step_count),
            "num_guidance_refreshes_blocked_by_gate": float(self.guidance_blocked_refresh_count),
            "num_bias_applications": float(self.bias_application_count),
            "num_noop_guidance_events": float(self.noop_guidance_events),
            "fallback_policy_used_count": float(self.fallback_event_count),
            "avg_num_affected_intersections": float(self.total_affected_intersections) / float(refresh_count),
            "avg_num_targeted_intersections": float(self.total_targeted_intersections) / float(refresh_count),
            "mean_bias_magnitude": float(self.total_bias_magnitude) / float(max(1, self.bias_application_count)),
            "max_bias_magnitude": float(self.max_bias_magnitude),
            "percent_steps_with_active_guidance": float(self.steps_with_active_guidance)
            / float(max(1, self.step_count)),
        }


@dataclass
class GuidedActionBatch:
    actions: np.ndarray
    base_actions: np.ndarray
    base_q_values: np.ndarray
    guided_q_values: np.ndarray
    q_bias: np.ndarray
    refresh_traces: list[GuidanceRefreshTrace] = field(default_factory=list)
    runtime_seconds: float = 0.0


class BaseGuidanceProvider(ABC):
    source_name: str

    @abstractmethod
    def generate(self, summary: DistrictStateSummary) -> GuidanceDecision:
        raise NotImplementedError


class HeuristicGuidanceProvider(BaseGuidanceProvider):
    source_name = "heuristic"

    def __init__(self, config: HeuristicGuidanceConfig | None = None):
        self.config = config or HeuristicGuidanceConfig()

    def generate(self, summary: DistrictStateSummary) -> GuidanceDecision:
        started = perf_counter()
        action = generate_heuristic_guidance(summary=summary, config=self.config)
        return GuidanceDecision(
            source=self.source_name,
            action=action,
            runtime_seconds=perf_counter() - started,
            parsed_payload_before_repair=action.to_dict(),
        )


class LLMGuidanceProvider(BaseGuidanceProvider):
    source_name = "llm"

    def __init__(self, inference: DistrictLLMInference, max_new_tokens: int = 128):
        self.inference = inference
        self.max_new_tokens = int(max_new_tokens)

    def generate(self, summary: DistrictStateSummary) -> GuidanceDecision:
        started = perf_counter()
        result: DistrictLLMInferenceResult = self.inference.predict_with_result(
            summary=summary,
            max_new_tokens=self.max_new_tokens,
        )
        return GuidanceDecision(
            source=self.source_name,
            action=result.action,
            runtime_seconds=perf_counter() - started,
            raw_text=result.raw_text,
            parsed_payload_before_repair=result.parsed_payload_before_repair,
            repair_report=result.repair_report,
            json_valid=result.json_valid,
            schema_valid_before_repair=result.schema_valid_before_repair,
        )


class FixedRLPolicyAdapter:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.teacher = RLCheckpointTeacher(checkpoint_path=checkpoint_path, device=device)
        self.device = self.teacher.device

    @property
    def env_config(self) -> Any | None:
        return self.teacher.env_config

    def decide(self, observation_batch: dict[str, Any]) -> RLPolicyDecision:
        raw_obs = observation_batch["observations"].astype(np.float32)
        normalized_obs = (
            self.teacher.obs_normalizer.normalize(raw_obs)
            if self.teacher.obs_normalizer is not None
            else raw_obs
        )
        obs_tensor = torch.as_tensor(normalized_obs, dtype=torch.float32, device=self.device)
        district_type_tensor = torch.as_tensor(
            observation_batch["district_type_indices"],
            dtype=torch.int64,
            device=self.device,
        )
        action_mask_tensor = torch.as_tensor(
            observation_batch["action_mask"],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            q_values = self.teacher.model.forward(
                observations=obs_tensor,
                district_type_indices=district_type_tensor,
                action_mask=action_mask_tensor,
            )
        q_values_np = q_values.detach().cpu().numpy().astype(np.float32)
        return RLPolicyDecision(q_values=q_values_np, actions=q_values_np.argmax(axis=1).astype(np.int64))


class DistrictGuidedRLController:
    def __init__(
        self,
        policy: FixedRLPolicyAdapter,
        mode_source: str,
        summary_builder: DistrictStateSummaryBuilder | None = None,
        guidance_provider: BaseGuidanceProvider | None = None,
        influence_config: GuidanceInfluenceConfig | None = None,
        heuristic_provider: BaseGuidanceProvider | None = None,
    ):
        self.policy = policy
        self.mode_source = mode_source
        self.summary_builder = summary_builder
        self.guidance_provider = guidance_provider
        self.influence_config = (influence_config or GuidanceInfluenceConfig()).validate()
        self.heuristic_provider = heuristic_provider
        self._active_guidance: dict[str, ActiveDistrictGuidance] = {}
        self._next_refresh_step_by_district: dict[str, int] = {}
        self._episode_stats = WrapperEpisodeStats()

    def reset(self) -> None:
        self._active_guidance = {}
        self._next_refresh_step_by_district = {}
        self._episode_stats = WrapperEpisodeStats()
        if self.summary_builder is not None:
            self.summary_builder.reset()

    def active_guidance_snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            district_id: active.decision.action.to_dict()
            for district_id, active in sorted(self._active_guidance.items())
        }

    def episode_debug_summary(self) -> dict[str, Any]:
        payload = self._episode_stats.to_dict()
        payload.update(
            {
                "wrapper_mode": self.influence_config.wrapper_mode,
                "fallback_policy": self.influence_config.fallback_policy,
            }
        )
        return payload

    def act(self, env, observation_batch: dict[str, Any]) -> GuidedActionBatch:
        started = perf_counter()
        base_decision = self.policy.decide(observation_batch)
        base_q_values = base_decision.q_values
        guided_q_values = base_q_values.copy()
        q_bias = np.zeros_like(guided_q_values, dtype=np.float32)

        refresh_traces = self._refresh_guidance_if_needed(env=env, observation_batch=observation_batch)
        if self.guidance_provider is None:
            self._episode_stats.step_count += 1
            return GuidedActionBatch(
                actions=base_decision.actions.copy(),
                base_actions=base_decision.actions,
                base_q_values=base_q_values,
                guided_q_values=guided_q_values,
                q_bias=q_bias,
                refresh_traces=refresh_traces,
                runtime_seconds=perf_counter() - started,
            )

        active_any = False
        decision_step = int(observation_batch.get("decision_step", 0))
        candidate_lookup_by_district = {
            district_id: {
                item.intersection_id: item
                for item in active.summary.candidate_intersections
            }
            for district_id, active in self._active_guidance.items()
        }
        for row_index, intersection_id in enumerate(observation_batch["intersection_ids"]):
            district_id = str(observation_batch["district_ids"][row_index])
            active = self._active_guidance.get(district_id)
            if active is None:
                continue
            active_any = True
            candidate = candidate_lookup_by_district[district_id].get(str(intersection_id))
            row_bias = self._row_action_bias(
                active=active,
                candidate=candidate,
                intersection_id=str(intersection_id),
                current_phase=int(observation_batch["current_phase"][row_index]),
                decision_step=decision_step,
            )
            if row_bias is None:
                continue
            q_bias[row_index] = row_bias
            guided_q_values[row_index] = guided_q_values[row_index] + row_bias
            magnitude = float(np.abs(row_bias).max())
            self._episode_stats.bias_application_count += 1
            self._episode_stats.total_bias_magnitude += magnitude
            self._episode_stats.max_bias_magnitude = max(self._episode_stats.max_bias_magnitude, magnitude)

        self._episode_stats.step_count += 1
        if active_any:
            self._episode_stats.steps_with_active_guidance += 1
        actions = guided_q_values.argmax(axis=1).astype(np.int64)
        return GuidedActionBatch(
            actions=actions,
            base_actions=base_decision.actions,
            base_q_values=base_q_values,
            guided_q_values=guided_q_values,
            q_bias=q_bias,
            refresh_traces=refresh_traces,
            runtime_seconds=perf_counter() - started,
        )

    def _refresh_guidance_if_needed(
        self,
        env,
        observation_batch: dict[str, Any],
    ) -> list[GuidanceRefreshTrace]:
        if self.guidance_provider is None or self.summary_builder is None:
            return []

        decision_step = int(observation_batch.get("decision_step", 0))
        due_districts = [
            district_id
            for district_id in tuple(sorted(env.districts))
            if self._district_requires_refresh(district_id=district_id, decision_step=decision_step)
        ]
        if not due_districts:
            return []

        summaries = self.summary_builder.build_all(env, observation_batch)
        refresh_traces: list[GuidanceRefreshTrace] = []
        gate_blocked_this_step = False
        for district_id in due_districts:
            summary = summaries[district_id]
            previous_active = self._active_guidance.get(district_id)
            gate_decision = _evaluate_guidance_gate(summary=summary, config=self.influence_config)
            if not gate_decision.allowed:
                gate_blocked_this_step = True
                self._active_guidance.pop(district_id, None)
                self._next_refresh_step_by_district[district_id] = (
                    decision_step + self._resolve_blocked_refresh_horizon()
                )
                self._episode_stats.guidance_refresh_count += 1
                self._episode_stats.guidance_blocked_refresh_count += 1
                self._episode_stats.noop_guidance_events += 1
                decision = GuidanceDecision(
                    source=f"{self.mode_source}_gate_blocked",
                    action=DistrictAction.default_hold(),
                    runtime_seconds=0.0,
                    fallback_policy_applied="gate_blocked",
                )
                application_plan = _build_application_plan(
                    summary=summary,
                    action=decision.action,
                    config=replace(self.influence_config, wrapper_mode="no_op"),
                    district_intersection_ids=tuple(env.districts[district_id].intersection_ids),
                )
                trace = GuidanceRefreshTrace(
                    mode_source=self.mode_source,
                    district_id=district_id,
                    decision_step=decision_step,
                    summary_hash=_summary_hash(summary),
                    summary_excerpt=summary.to_prompt_text()[:240],
                    summary_payload=summary.to_dict(),
                    guidance=decision.to_trace_payload(),
                    repaired_guidance=decision.action.to_dict(),
                    fallback_used=False,
                    fallback_policy="gate_blocked",
                    application_plan=application_plan.to_dict(),
                    applied_biases={
                        "base": 0.0,
                        "target": 0.0,
                        "corridor": 0.0,
                        "strength_scale": 0.0,
                    },
                    gate_decision=gate_decision.to_dict(),
                )
                refresh_traces.append(trace)
                if self.influence_config.log_guidance_debug:
                    _log_guidance_debug(trace)
                continue

            decision, fallback_used = self._generate_guidance(
                district_id=district_id,
                summary=summary,
                previous_active=previous_active,
            )
            application_plan = _build_application_plan(
                summary=summary,
                action=decision.action,
                config=self.influence_config,
                district_intersection_ids=tuple(env.districts[district_id].intersection_ids),
            )
            expires_step = decision_step + self._resolve_refresh_horizon(decision.action)
            active = ActiveDistrictGuidance(
                district_id=district_id,
                summary=summary,
                decision=decision,
                application_plan=application_plan,
                generated_step=decision_step,
                expires_step=expires_step,
                fallback_used=fallback_used,
            )
            self._active_guidance[district_id] = active
            self._next_refresh_step_by_district[district_id] = int(expires_step)

            self._episode_stats.guidance_refresh_count += 1
            self._episode_stats.total_affected_intersections += len(application_plan.affected_intersections)
            self._episode_stats.total_targeted_intersections += len(application_plan.targeted_intersections)
            if application_plan.wrapper_mode == "no_op" or not application_plan.affected_intersections:
                self._episode_stats.noop_guidance_events += 1
            if fallback_used:
                self._episode_stats.fallback_event_count += 1

            trace = GuidanceRefreshTrace(
                mode_source=self.mode_source,
                district_id=district_id,
                decision_step=decision_step,
                summary_hash=_summary_hash(summary),
                summary_excerpt=summary.to_prompt_text()[:240],
                summary_payload=summary.to_dict(),
                guidance=decision.to_trace_payload(),
                repaired_guidance=decision.action.to_dict(),
                fallback_used=fallback_used,
                fallback_policy=self.influence_config.fallback_policy if fallback_used else "none",
                application_plan=application_plan.to_dict(),
                applied_biases={
                    "base": float(application_plan.base_bias_strength),
                    "target": float(application_plan.target_bias_strength),
                    "corridor": float(application_plan.corridor_bias_strength),
                    "strength_scale": float(application_plan.strength_scale),
                },
                gate_decision=gate_decision.to_dict(),
            )
            refresh_traces.append(trace)
            if self.influence_config.log_guidance_debug:
                _log_guidance_debug(trace)
        if gate_blocked_this_step:
            self._episode_stats.guidance_blocked_step_count += 1
        return refresh_traces

    def _generate_guidance(
        self,
        district_id: str,
        summary: DistrictStateSummary,
        previous_active: ActiveDistrictGuidance | None,
    ) -> tuple[GuidanceDecision, bool]:
        fallback_used = False
        try:
            decision = self.guidance_provider.generate(summary)
        except Exception as exc:
            decision = GuidanceDecision(
                source=self.guidance_provider.source_name,
                action=DistrictAction.default_hold(),
                runtime_seconds=0.0,
                provider_error=str(exc),
                json_valid=False,
                schema_valid_before_repair=False,
            )
        if not _should_fallback(decision):
            return decision, fallback_used

        fallback_used = True
        fallback_policy = self.influence_config.fallback_policy
        if fallback_policy == "hold_previous" and previous_active is not None:
            fallback_decision = GuidanceDecision(
                source=f"{decision.source}_fallback_hold_previous",
                action=previous_active.decision.action,
                runtime_seconds=decision.runtime_seconds,
                raw_text=decision.raw_text,
                parsed_payload_before_repair=decision.parsed_payload_before_repair,
                repair_report=decision.repair_report,
                json_valid=decision.json_valid,
                schema_valid_before_repair=decision.schema_valid_before_repair,
                provider_error=decision.provider_error,
                fallback_policy_applied=fallback_policy,
            )
            return fallback_decision, fallback_used

        if fallback_policy == "heuristic_weak" and self.heuristic_provider is not None:
            fallback_decision = self.heuristic_provider.generate(summary)
            fallback_decision.fallback_policy_applied = fallback_policy
            return fallback_decision, fallback_used

        fallback_decision = GuidanceDecision(
            source=f"{decision.source}_fallback_no_op",
            action=DistrictAction.default_hold(),
            runtime_seconds=decision.runtime_seconds,
            raw_text=decision.raw_text,
            parsed_payload_before_repair=decision.parsed_payload_before_repair,
            repair_report=decision.repair_report,
            json_valid=decision.json_valid,
            schema_valid_before_repair=decision.schema_valid_before_repair,
            provider_error=decision.provider_error,
            fallback_policy_applied=fallback_policy,
        )
        return fallback_decision, fallback_used

    def _district_requires_refresh(self, district_id: str, decision_step: int) -> bool:
        next_refresh_step = self._next_refresh_step_by_district.get(district_id)
        if next_refresh_step is None:
            return True
        return decision_step >= int(next_refresh_step)

    def _resolve_refresh_horizon(self, action: DistrictAction) -> int:
        requested = max(1, min(int(action.duration_steps), self.influence_config.max_guidance_duration))
        return min(
            requested,
            int(self.influence_config.guidance_refresh_steps),
            int(self.influence_config.guidance_persistence_steps),
        )

    def _resolve_blocked_refresh_horizon(self) -> int:
        return max(
            1,
            min(
                int(self.influence_config.guidance_refresh_steps),
                int(self.influence_config.guidance_persistence_steps),
            ),
        )

    def _row_action_bias(
        self,
        active: ActiveDistrictGuidance,
        candidate: CandidateIntersection | None,
        intersection_id: str,
        current_phase: int,
        decision_step: int,
    ) -> np.ndarray | None:
        plan = active.application_plan
        if plan.wrapper_mode == "no_op":
            return None
        if intersection_id not in set(plan.affected_intersections):
            return None

        preferred_action = _preferred_action_for_direction(
            direction=plan.priority_direction,
            current_phase=current_phase,
        )
        if preferred_action is None:
            return None

        decay = 1.0
        if self.influence_config.enable_bias_decay:
            horizon = max(1, active.expires_step - active.generated_step)
            age = max(0, decision_step - active.generated_step)
            if self.influence_config.bias_decay_schedule == "linear":
                decay = max(0.25, 1.0 - (float(age) / float(horizon)))

        magnitude = plan.base_bias_strength * plan.strength_scale * decay
        if intersection_id in set(plan.targeted_intersections):
            magnitude += plan.target_bias_strength * plan.strength_scale * decay
        if candidate is not None and plan.priority_direction in {"NS", "EW"}:
            if candidate.corridor_alignment == plan.priority_direction:
                magnitude += plan.corridor_bias_strength * plan.strength_scale * decay
            if candidate.is_boundary and plan.scope in {"corridor_local", "global"}:
                magnitude += 0.5 * plan.corridor_bias_strength * plan.strength_scale * decay

        strategy_multiplier = STRATEGY_BIAS_MULTIPLIERS.get(active.decision.action.strategy, 1.0)
        magnitude *= strategy_multiplier
        if magnitude <= 0.0:
            return None

        bias = np.zeros(2, dtype=np.float32)
        bias[preferred_action] += float(magnitude)
        return bias


def _build_application_plan(
    summary: DistrictStateSummary,
    action: DistrictAction,
    config: GuidanceInfluenceConfig,
    district_intersection_ids: tuple[str, ...],
) -> GuidanceApplicationPlan:
    wrapper_mode = config.wrapper_mode
    target_ids = tuple(
        intersection_id
        for intersection_id in action.target_intersections
        if intersection_id in {item.intersection_id for item in summary.candidate_intersections}
    )
    candidate_lookup = {
        item.intersection_id: item
        for item in summary.candidate_intersections
    }
    priority_direction = _resolve_guidance_direction(action=action, summary=summary)
    if wrapper_mode == "no_op":
        return GuidanceApplicationPlan(
            wrapper_mode=wrapper_mode,
            scope="none",
            affected_intersections=(),
            targeted_intersections=target_ids,
            target_candidate_ids=tuple(candidate_lookup),
            priority_direction=priority_direction,
            strength_scale=0.0,
            base_bias_strength=0.0,
            target_bias_strength=0.0,
            corridor_bias_strength=0.0,
            apply_global_bias=False,
            apply_target_only=True,
            max_intersections_affected=0,
        )

    if wrapper_mode == "current_legacy":
        affected = tuple(district_intersection_ids)
        return GuidanceApplicationPlan(
            wrapper_mode=wrapper_mode,
            scope="global",
            affected_intersections=affected,
            targeted_intersections=target_ids,
            target_candidate_ids=tuple(candidate_lookup),
            priority_direction=priority_direction,
            strength_scale=1.0,
            base_bias_strength=float(max(config.bias_strength, 0.75)),
            target_bias_strength=float(max(config.target_only_bias_strength, 1.25)),
            corridor_bias_strength=float(max(config.corridor_bias_strength, 0.5)),
            apply_global_bias=True,
            apply_target_only=False,
            max_intersections_affected=max(len(affected), config.max_intersections_affected),
        )

    if wrapper_mode in {"target_only_soft", "target_only_medium"}:
        strength_scale = 0.5 if wrapper_mode == "target_only_soft" else 1.0
        affected = target_ids[: config.max_intersections_affected]
        return GuidanceApplicationPlan(
            wrapper_mode=wrapper_mode,
            scope="targeted",
            affected_intersections=affected,
            targeted_intersections=target_ids,
            target_candidate_ids=tuple(candidate_lookup),
            priority_direction=priority_direction,
            strength_scale=strength_scale,
            base_bias_strength=float(config.bias_strength),
            target_bias_strength=float(config.target_only_bias_strength),
            corridor_bias_strength=float(config.corridor_bias_strength),
            apply_global_bias=False,
            apply_target_only=True,
            max_intersections_affected=config.max_intersections_affected,
        )

    if wrapper_mode == "corridor_soft":
        ranked = list(target_ids)
        extras = [
            item.intersection_id
            for item in summary.candidate_intersections
            if item.intersection_id not in ranked
            and item.is_boundary
            and (priority_direction is None or item.corridor_alignment == priority_direction)
        ]
        affected = tuple((ranked + extras)[: config.max_intersections_affected])
        return GuidanceApplicationPlan(
            wrapper_mode=wrapper_mode,
            scope="corridor_local",
            affected_intersections=affected,
            targeted_intersections=target_ids,
            target_candidate_ids=tuple(candidate_lookup),
            priority_direction=priority_direction,
            strength_scale=0.6,
            base_bias_strength=float(config.bias_strength),
            target_bias_strength=float(config.target_only_bias_strength),
            corridor_bias_strength=float(config.corridor_bias_strength),
            apply_global_bias=False,
            apply_target_only=False,
            max_intersections_affected=config.max_intersections_affected,
        )

    affected_global = tuple(district_intersection_ids)
    return GuidanceApplicationPlan(
        wrapper_mode="global_soft",
        scope="global",
        affected_intersections=affected_global,
        targeted_intersections=target_ids,
        target_candidate_ids=tuple(candidate_lookup),
        priority_direction=priority_direction,
        strength_scale=0.35,
        base_bias_strength=float(config.bias_strength),
        target_bias_strength=float(config.target_only_bias_strength),
        corridor_bias_strength=float(config.corridor_bias_strength),
        apply_global_bias=True,
        apply_target_only=False,
        max_intersections_affected=config.max_intersections_affected,
    )


def _should_fallback(decision: GuidanceDecision) -> bool:
    if decision.provider_error is not None:
        return True
    if not decision.json_valid or not decision.schema_valid_before_repair:
        return True
    report = decision.repair_report
    if report is None:
        return False
    return bool(
        report.fallback_used
        or report.empty_after_filtering
    )


def _evaluate_guidance_gate(
    summary: DistrictStateSummary,
    config: GuidanceInfluenceConfig,
) -> GuidanceGateDecision:
    queue_imbalance = abs(float(summary.ns_queue) - float(summary.ew_queue))
    queue_trigger = float(summary.avg_queue) >= float(config.min_avg_queue_for_guidance)
    imbalance_trigger = queue_imbalance >= float(config.min_queue_imbalance_for_guidance)
    incident_or_spillback = bool(summary.incident_flag or summary.spillback_risk or summary.overload_flag)
    triggers = {
        "incident_or_spillback": incident_or_spillback,
        "queue_threshold": queue_trigger,
        "imbalance_threshold": imbalance_trigger,
    }
    triggered_conditions = tuple(name for name, active in triggers.items() if active)

    if config.gating_mode == "always_on":
        allowed = True
    elif config.gating_mode == "incident_or_spillback":
        allowed = incident_or_spillback
    elif config.gating_mode == "queue_threshold":
        allowed = queue_trigger
    elif config.gating_mode == "imbalance_threshold":
        allowed = imbalance_trigger
    elif config.gating_mode == "queue_or_imbalance":
        allowed = queue_trigger or imbalance_trigger
    else:
        allowed = incident_or_spillback or queue_trigger or imbalance_trigger

    blocked_reasons: list[str] = []
    if config.require_incident_or_spillback and not incident_or_spillback:
        allowed = False
        blocked_reasons.append("requires_incident_or_spillback")
    if not config.allow_guidance_in_normal_conditions and not triggered_conditions:
        allowed = False
        blocked_reasons.append("normal_conditions_blocked")
    if not allowed and not blocked_reasons:
        blocked_reasons.append(f"gating_mode:{config.gating_mode}")

    return GuidanceGateDecision(
        allowed=allowed,
        gating_mode=config.gating_mode,
        triggered_conditions=triggered_conditions,
        blocked_reasons=tuple(blocked_reasons),
        avg_queue=float(summary.avg_queue),
        queue_imbalance=float(queue_imbalance),
        incident_flag=bool(summary.incident_flag),
        spillback_risk=bool(summary.spillback_risk),
        overload_flag=bool(summary.overload_flag),
    )


def _resolve_guidance_direction(action: DistrictAction, summary: DistrictStateSummary) -> str | None:
    if action.phase_bias in {"NS", "EW"}:
        return action.phase_bias
    if action.priority_corridor in {"NS", "EW"}:
        return action.priority_corridor
    if summary.dominant_flow in {"NS", "EW"}:
        return summary.dominant_flow
    return None


def _preferred_action_for_direction(direction: str | None, current_phase: int) -> int | None:
    if direction == "NS":
        return 0 if current_phase == 0 else 1
    if direction == "EW":
        return 0 if current_phase != 0 else 1
    return None


def _summary_hash(summary: DistrictStateSummary) -> str:
    return hashlib.sha1(summary.to_json().encode("utf-8")).hexdigest()[:16]


def guidance_config_payload(config: GuidanceInfluenceConfig) -> dict[str, Any]:
    return asdict(config.validate())


def _log_guidance_debug(trace: GuidanceRefreshTrace) -> None:
    print(
        "[guidance-debug] "
        f"mode={trace.mode_source} "
        f"district={trace.district_id} "
        f"wrapper_mode={trace.application_plan['wrapper_mode']} "
        f"gate_allowed={trace.gate_decision.get('allowed') if trace.gate_decision else True} "
        f"scope={trace.application_plan['scope']} "
        f"targets={trace.repaired_guidance.get('target_intersections', [])} "
        f"affected={trace.application_plan['affected_intersections']} "
        f"fallback_used={trace.fallback_used} "
        f"fallback_policy={trace.fallback_policy}"
    )
