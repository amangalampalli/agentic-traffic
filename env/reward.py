from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REWARD_VARIANTS: tuple[str, ...] = (
    "current",
    "normalized_wait_queue",
    "wait_queue_throughput",
)


@dataclass(frozen=True)
class RewardConfig:
    variant: str = "current"
    waiting_weight: float = 1.0
    vehicle_weight: float = 0.1
    pressure_weight: float = 0.0
    reward_scale: float = 0.1
    normalize_by_lane_count: bool = True
    clip_reward: float | None = 5.0
    queue_delta_weight: float = 2.0
    wait_delta_weight: float = 4.0
    queue_level_weight: float = 0.5
    wait_level_weight: float = 1.0
    throughput_weight: float = 0.1
    imbalance_weight: float = 0.1
    delta_clip: float = 2.0
    level_normalizer: float = 10.0
    throughput_normalizer: float = 2.0


@dataclass(frozen=True)
class RewardBreakdown:
    reward: np.ndarray
    components: dict[str, np.ndarray]


class RewardCalculator:
    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        if self.config.variant not in REWARD_VARIANTS:
            raise ValueError(
                f"Unsupported reward variant: {self.config.variant}. "
                f"Expected one of {REWARD_VARIANTS}."
            )
        self._prev_queue_norm: np.ndarray | None = None
        self._prev_wait_norm: np.ndarray | None = None
        self._prev_finished_vehicle_count = 0.0

    def reset(
        self,
        incoming_waiting: np.ndarray,
        incoming_counts: np.ndarray,
        incoming_lane_counts: np.ndarray | None = None,
        finished_vehicle_count: float = 0.0,
    ) -> None:
        queue_norm, wait_norm, _ = self._normalized_state(
            incoming_waiting=incoming_waiting,
            incoming_counts=incoming_counts,
            incoming_lane_counts=incoming_lane_counts,
        )
        self._prev_queue_norm = queue_norm
        self._prev_wait_norm = wait_norm
        self._prev_finished_vehicle_count = float(finished_vehicle_count)

    def compute(
        self,
        incoming_waiting: np.ndarray,
        incoming_counts: np.ndarray,
        outgoing_counts: np.ndarray | None = None,
        incoming_lane_counts: np.ndarray | None = None,
        finished_vehicle_count: float = 0.0,
    ) -> np.ndarray:
        return self.compute_breakdown(
            incoming_waiting=incoming_waiting,
            incoming_counts=incoming_counts,
            outgoing_counts=outgoing_counts,
            incoming_lane_counts=incoming_lane_counts,
            finished_vehicle_count=finished_vehicle_count,
        ).reward

    def compute_breakdown(
        self,
        incoming_waiting: np.ndarray,
        incoming_counts: np.ndarray,
        outgoing_counts: np.ndarray | None = None,
        incoming_lane_counts: np.ndarray | None = None,
        finished_vehicle_count: float = 0.0,
    ) -> RewardBreakdown:
        if self.config.variant == "current":
            return self._compute_current(
                incoming_waiting=incoming_waiting,
                incoming_counts=incoming_counts,
                outgoing_counts=outgoing_counts,
                incoming_lane_counts=incoming_lane_counts,
            )
        return self._compute_delta_based(
            incoming_waiting=incoming_waiting,
            incoming_counts=incoming_counts,
            incoming_lane_counts=incoming_lane_counts,
            finished_vehicle_count=finished_vehicle_count,
            include_throughput=self.config.variant == "wait_queue_throughput",
        )

    def _compute_current(
        self,
        incoming_waiting: np.ndarray,
        incoming_counts: np.ndarray,
        outgoing_counts: np.ndarray | None = None,
        incoming_lane_counts: np.ndarray | None = None,
    ) -> RewardBreakdown:
        waiting_total = incoming_waiting.sum(axis=1)
        vehicle_total = incoming_counts.sum(axis=1)
        normalization = self._lane_normalization(waiting_total.shape[0], incoming_lane_counts)

        components = {
            "wait_term": (-self.config.waiting_weight * waiting_total / normalization).astype(np.float32),
            "queue_term": (-self.config.vehicle_weight * vehicle_total / normalization).astype(np.float32),
        }
        if outgoing_counts is not None and self.config.pressure_weight != 0.0:
            outgoing_total = outgoing_counts.sum(axis=1)
            components["pressure_term"] = (
                self.config.pressure_weight * (outgoing_total - vehicle_total) / normalization
            ).astype(np.float32)
        components = self._scale_components(components)
        reward = self._finalize_reward(components)
        return RewardBreakdown(reward=reward, components=components)

    def _compute_delta_based(
        self,
        incoming_waiting: np.ndarray,
        incoming_counts: np.ndarray,
        incoming_lane_counts: np.ndarray | None,
        finished_vehicle_count: float,
        include_throughput: bool,
    ) -> RewardBreakdown:
        queue_norm, wait_norm, lane_norm = self._normalized_state(
            incoming_waiting=incoming_waiting,
            incoming_counts=incoming_counts,
            incoming_lane_counts=incoming_lane_counts,
        )

        if self._prev_queue_norm is None or self._prev_wait_norm is None:
            self._prev_queue_norm = queue_norm.copy()
            self._prev_wait_norm = wait_norm.copy()

        queue_delta = np.clip(
            self._prev_queue_norm - queue_norm,
            -self.config.delta_clip,
            self.config.delta_clip,
        ).astype(np.float32)
        wait_delta = np.clip(
            self._prev_wait_norm - wait_norm,
            -self.config.delta_clip,
            self.config.delta_clip,
        ).astype(np.float32)

        components: dict[str, np.ndarray] = {
            "queue_term": (self.config.queue_delta_weight * queue_delta).astype(np.float32),
            "wait_term": (self.config.wait_delta_weight * wait_delta).astype(np.float32),
            "queue_level_term": (
                -self.config.queue_level_weight
                * np.clip(queue_norm / self.config.level_normalizer, 0.0, self.config.delta_clip)
            ).astype(np.float32),
            "wait_level_term": (
                -self.config.wait_level_weight
                * np.clip(wait_norm / self.config.level_normalizer, 0.0, self.config.delta_clip)
            ).astype(np.float32),
        }

        if include_throughput:
            num_intersections = max(1, queue_norm.shape[0])
            finished_delta = max(
                0.0,
                float(finished_vehicle_count) - self._prev_finished_vehicle_count,
            )
            throughput_per_intersection = finished_delta / float(num_intersections)
            throughput_term = np.full(
                queue_norm.shape,
                self.config.throughput_weight
                * min(1.0, throughput_per_intersection / self.config.throughput_normalizer),
                dtype=np.float32,
            )
            imbalance = np.std(
                incoming_waiting / lane_norm[:, None],
                axis=1,
            ).astype(np.float32)
            components["throughput_term"] = throughput_term
            components["imbalance_term"] = (-self.config.imbalance_weight * imbalance).astype(
                np.float32
            )

        components = self._scale_components(components)
        reward = self._finalize_reward(components)
        self._prev_queue_norm = queue_norm
        self._prev_wait_norm = wait_norm
        self._prev_finished_vehicle_count = float(finished_vehicle_count)
        return RewardBreakdown(reward=reward, components=components)

    def _normalized_state(
        self,
        incoming_waiting: np.ndarray,
        incoming_counts: np.ndarray,
        incoming_lane_counts: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lane_norm = self._lane_normalization(incoming_counts.shape[0], incoming_lane_counts)
        queue_norm = (incoming_counts.sum(axis=1) / lane_norm).astype(np.float32)
        wait_norm = (incoming_waiting.sum(axis=1) / lane_norm).astype(np.float32)
        return queue_norm, wait_norm, lane_norm

    def _lane_normalization(
        self,
        batch_size: int,
        incoming_lane_counts: np.ndarray | None,
    ) -> np.ndarray:
        normalization = np.ones(batch_size, dtype=np.float32)
        if incoming_lane_counts is not None and self.config.normalize_by_lane_count:
            normalization = np.maximum(1.0, incoming_lane_counts.astype(np.float32))
        return normalization

    def _finalize_reward(self, components: dict[str, np.ndarray]) -> np.ndarray:
        reward = np.zeros_like(next(iter(components.values())), dtype=np.float32)
        for term in components.values():
            reward += term.astype(np.float32)

        if self.config.clip_reward is not None:
            reward = np.clip(
                reward,
                -float(self.config.clip_reward),
                float(self.config.clip_reward),
            )
        return reward.astype(np.float32)

    def _scale_components(
        self,
        components: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        scale = float(self.config.reward_scale)
        return {
            name: (values.astype(np.float32) * scale).astype(np.float32)
            for name, values in components.items()
        }
