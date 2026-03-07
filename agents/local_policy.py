from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseLocalPolicy(ABC):
    @abstractmethod
    def act(self, observation_batch: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class HoldPhasePolicy(BaseLocalPolicy):
    def act(self, observation_batch: dict[str, np.ndarray]) -> np.ndarray:
        intersection_count = len(observation_batch["intersection_ids"])
        return np.zeros(intersection_count, dtype=np.int64)


class RandomPhasePolicy(BaseLocalPolicy):
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def act(self, observation_batch: dict[str, np.ndarray]) -> np.ndarray:
        action_mask = observation_batch["action_mask"]
        actions = np.zeros(action_mask.shape[0], dtype=np.int64)
        for row_index, mask in enumerate(action_mask):
            valid_actions = np.flatnonzero(mask > 0.0)
            actions[row_index] = int(self.rng.choice(valid_actions))
        return actions


class FixedCyclePolicy(BaseLocalPolicy):
    def __init__(self, green_time: int = 20):
        self.green_time = int(green_time)

    def act(self, observation_batch: dict[str, np.ndarray]) -> np.ndarray:
        elapsed = observation_batch["phase_elapsed"]
        action_mask = observation_batch["action_mask"]
        should_switch = (elapsed >= self.green_time) & (action_mask[:, 1] > 0.0)
        return should_switch.astype(np.int64)


class QueueGreedyPolicy(BaseLocalPolicy):
    def __init__(self, switch_margin: float = 1.0):
        self.switch_margin = float(switch_margin)

    def act(self, observation_batch: dict[str, np.ndarray]) -> np.ndarray:
        counts = observation_batch["incoming_counts"]
        waiting = observation_batch["incoming_waiting"]
        lane_mask = observation_batch["lane_mask"]
        current_phase = observation_batch["current_phase"]
        action_mask = observation_batch["action_mask"]

        midpoint = counts.shape[1] // 2
        ns_score = (
            counts[:, :midpoint].sum(axis=1)
            + 1.5 * waiting[:, :midpoint].sum(axis=1)
        )
        ew_score = (
            counts[:, midpoint:].sum(axis=1)
            + 1.5 * waiting[:, midpoint:].sum(axis=1)
        )

        valid_midpoint = lane_mask[:, :midpoint].sum(axis=1) > 0
        ns_score = np.where(valid_midpoint, ns_score, 0.0)

        desired_switch = np.where(
            current_phase == 0,
            ew_score > ns_score + self.switch_margin,
            ns_score > ew_score + self.switch_margin,
        )
        desired_switch = desired_switch & (action_mask[:, 1] > 0.0)
        return desired_switch.astype(np.int64)


class SharedHeuristicLocalPolicy(QueueGreedyPolicy):
    def __init__(
        self,
        min_green_steps: int = 5,
        switch_margin: float = 1.0,
        district_bonus_scale: float = 0.0,
        neighbor_pressure_scale: float = 0.0,
    ):
        self.min_green_steps = int(min_green_steps)
        del district_bonus_scale, neighbor_pressure_scale
        super().__init__(switch_margin=switch_margin)

    def act_batch(self, observation_batch):
        if "intersection_ids" in observation_batch:
            return self.act(observation_batch)

        actions: dict[str, int] = {}
        for intersection_id, payload in observation_batch.items():
            waiting = payload.get("waiting_counts", [0, 0, 0, 0])
            queues = payload.get("queue_lengths", [0, 0, 0, 0])
            current_phase = int(payload.get("current_phase", 0))
            time_since_switch = int(payload.get("time_since_switch", 0))

            ns_score = float(sum(queues[:2]) + 1.5 * sum(waiting[:2]))
            ew_score = float(sum(queues[2:4]) + 1.5 * sum(waiting[2:4]))
            desired_phase = 0 if ns_score >= ew_score else 1

            if time_since_switch < self.min_green_steps:
                actions[intersection_id] = current_phase
            elif desired_phase != current_phase and abs(ns_score - ew_score) <= self.switch_margin:
                actions[intersection_id] = current_phase
            else:
                actions[intersection_id] = desired_phase
        return actions
