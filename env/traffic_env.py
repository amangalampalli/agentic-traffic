from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from env.cityflow_adapter import CityFlowAdapter
from env.intersection_config import DistrictConfig, IntersectionConfig
from env.observation_builder import ObservationBuilder, ObservationConfig
from env.reward import RewardCalculator, RewardConfig
from env.utils import build_topology, load_json


@dataclass(frozen=True)
class EnvConfig:
    simulator_interval: int = 1
    decision_interval: int = 5
    min_green_time: int = 10
    thread_num: int = 1
    observation: ObservationConfig = ObservationConfig()
    reward: RewardConfig = RewardConfig()
    max_episode_seconds: int | None = None


class TrafficEnv:
    def __init__(
        self,
        city_id: str,
        scenario_name: str,
        city_dir: str | Path,
        scenario_dir: str | Path,
        config_path: str | Path,
        roadnet_path: str | Path,
        district_map_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        env_config: EnvConfig | None = None,
    ):
        self.city_id = city_id
        self.scenario_name = scenario_name
        self.city_dir = Path(city_dir)
        self.scenario_dir = Path(scenario_dir)
        self.original_config_path = Path(config_path)
        self.roadnet_path = Path(roadnet_path)
        self.district_map_path = Path(district_map_path) if district_map_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.env_config = env_config or EnvConfig()

        self.intersections, self.districts = build_topology(
            roadnet_path=self.roadnet_path,
            district_map_path=self.district_map_path,
            metadata_path=self.metadata_path,
        )
        if not self.intersections:
            raise ValueError(
                f"No controllable intersections found for {self.city_id}/{self.scenario_name}."
            )

        self.controlled_intersection_ids = tuple(sorted(self.intersections))
        self.observation_builder = ObservationBuilder(
            intersections=self.intersections,
            districts=self.districts,
            config=self.env_config.observation,
        )
        self.reward_calculator = RewardCalculator(self.env_config.reward)
        self.adapter = CityFlowAdapter(
            config_path=self.original_config_path,
            thread_num=self.env_config.thread_num,
        )

        config_payload = load_json(self.original_config_path)
        self.max_episode_seconds = int(
            self.env_config.max_episode_seconds
            or config_payload.get("step", 0)
        )
        self.metadata = load_json(self.metadata_path) if self.metadata_path else {}
        self._district_type_labels = tuple(
            self.intersections[intersection_id].district_type
            for intersection_id in self.controlled_intersection_ids
        )
        self._incoming_lane_counts = np.asarray(
            [
                max(1, len(self.intersections[intersection_id].incoming_lanes))
                for intersection_id in self.controlled_intersection_ids
            ],
            dtype=np.float32,
        )

        self.current_phase_positions: dict[str, int] = {}
        self.phase_elapsed_times: dict[str, int] = {}
        self.decision_step_count = 0
        self.episode_return = 0.0
        self.total_episode_return = 0.0
        self.last_info: dict[str, Any] = {}
        self.reward_component_sums: dict[str, float] = {}

    @property
    def observation_dim(self) -> int:
        return self.observation_builder.observation_dim

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        del seed
        self.adapter.reset()
        self.decision_step_count = 0
        self.episode_return = 0.0
        self.total_episode_return = 0.0
        self.reward_component_sums = {}

        self.current_phase_positions = {}
        self.phase_elapsed_times = {}
        for intersection_id in self.controlled_intersection_ids:
            config = self.intersections[intersection_id]
            initial_position = 0
            initial_phase = config.green_phases[initial_position].engine_phase_index
            self.current_phase_positions[intersection_id] = initial_position
            self.phase_elapsed_times[intersection_id] = 0
            self.adapter.set_tl_phase(intersection_id, initial_phase)

        observation = self._build_observation()
        self.reward_calculator.reset(
            incoming_waiting=observation["incoming_waiting"],
            incoming_counts=observation["incoming_counts"],
            incoming_lane_counts=self._incoming_lane_counts,
            finished_vehicle_count=self.adapter.get_finished_vehicle_count(),
        )
        self.last_info = self._build_info(
            rewards=np.zeros(len(self.controlled_intersection_ids), dtype=np.float32),
            avg_incoming_counts=observation["incoming_counts"],
            avg_incoming_waiting=observation["incoming_waiting"],
            reward_components={},
        )
        return observation

    def step(
        self,
        actions: dict[str, int] | list[int] | np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray, bool, dict[str, Any]]:
        normalized_actions = self._normalize_actions(actions)
        self._apply_actions(normalized_actions)

        avg_incoming_counts, avg_incoming_waiting, avg_outgoing_counts = self._advance_simulator()
        reward_breakdown = self.reward_calculator.compute_breakdown(
            incoming_waiting=avg_incoming_waiting,
            incoming_counts=avg_incoming_counts,
            outgoing_counts=avg_outgoing_counts,
            incoming_lane_counts=self._incoming_lane_counts,
            finished_vehicle_count=self.adapter.get_finished_vehicle_count(),
        )
        rewards = reward_breakdown.reward
        self.decision_step_count += 1
        self.total_episode_return += float(rewards.sum())
        self.episode_return = self._mean_step_intersection_reward()
        self._accumulate_reward_components(reward_breakdown.components)

        observation = self._build_observation()
        done = self.adapter.get_current_time() >= self.max_episode_seconds
        info = self._build_info(
            rewards=rewards,
            avg_incoming_counts=avg_incoming_counts,
            avg_incoming_waiting=avg_incoming_waiting,
            reward_components=reward_breakdown.components,
        )
        self.last_info = info
        return observation, rewards, done, info

    def _build_observation(self) -> dict[str, Any]:
        lane_vehicle_count = self.adapter.get_lane_vehicle_count()
        lane_waiting_count = self.adapter.get_lane_waiting_vehicle_count()
        switch_allowed = {
            intersection_id: (
                self.phase_elapsed_times[intersection_id] >= self.env_config.min_green_time
            )
            for intersection_id in self.controlled_intersection_ids
        }

        observation = self.observation_builder.build(
            lane_vehicle_count=lane_vehicle_count,
            lane_waiting_count=lane_waiting_count,
            phase_positions=self.current_phase_positions,
            phase_elapsed_times=self.phase_elapsed_times,
            switch_allowed=switch_allowed,
        )
        observation["city_id"] = self.city_id
        observation["scenario_name"] = self.scenario_name
        observation["decision_step"] = self.decision_step_count
        observation["sim_time"] = self.adapter.get_current_time()
        return observation

    def _apply_actions(self, actions: np.ndarray) -> None:
        for action_index, intersection_id in enumerate(self.controlled_intersection_ids):
            config = self.intersections[intersection_id]
            current_position = self.current_phase_positions[intersection_id]
            can_switch = self.phase_elapsed_times[intersection_id] >= self.env_config.min_green_time
            should_switch = int(actions[action_index]) == 1 and can_switch

            if should_switch:
                next_position = (current_position + 1) % config.num_green_phases
                engine_phase = config.green_phases[next_position].engine_phase_index
                self.adapter.set_tl_phase(intersection_id, engine_phase)
                self.current_phase_positions[intersection_id] = next_position
                self.phase_elapsed_times[intersection_id] = 0
            else:
                current_engine_phase = config.green_phases[current_position].engine_phase_index
                self.adapter.set_tl_phase(intersection_id, current_engine_phase)

    def _advance_simulator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_intersections = len(self.controlled_intersection_ids)
        max_lanes = self.env_config.observation.max_incoming_lanes
        avg_incoming_counts = np.zeros((num_intersections, max_lanes), dtype=np.float32)
        avg_incoming_waiting = np.zeros((num_intersections, max_lanes), dtype=np.float32)
        avg_outgoing_counts = np.zeros((num_intersections, max_lanes), dtype=np.float32)

        for _ in range(self.env_config.decision_interval):
            self.adapter.step()
            lane_vehicle_count = self.adapter.get_lane_vehicle_count()
            lane_waiting_count = self.adapter.get_lane_waiting_vehicle_count()

            for row_index, intersection_id in enumerate(self.controlled_intersection_ids):
                config = self.intersections[intersection_id]
                for lane_index, lane_id in enumerate(
                    config.incoming_lanes[: self.env_config.observation.max_incoming_lanes]
                ):
                    avg_incoming_counts[row_index, lane_index] += float(
                        lane_vehicle_count.get(lane_id, 0)
                    )
                    avg_incoming_waiting[row_index, lane_index] += float(
                        lane_waiting_count.get(lane_id, 0)
                    )
                for lane_index, lane_id in enumerate(
                    config.outgoing_lanes[: self.env_config.observation.max_incoming_lanes]
                ):
                    avg_outgoing_counts[row_index, lane_index] += float(
                        lane_vehicle_count.get(lane_id, 0)
                    )

                self.phase_elapsed_times[intersection_id] += self.env_config.simulator_interval

        avg_incoming_counts /= float(self.env_config.decision_interval)
        avg_incoming_waiting /= float(self.env_config.decision_interval)
        avg_outgoing_counts /= float(self.env_config.decision_interval)
        return avg_incoming_counts, avg_incoming_waiting, avg_outgoing_counts

    def _build_info(
        self,
        rewards: np.ndarray,
        avg_incoming_counts: np.ndarray,
        avg_incoming_waiting: np.ndarray,
        reward_components: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        mean_reward = float(rewards.mean()) if rewards.size else 0.0
        average_travel_time = self.adapter.get_average_travel_time()
        info = {
            "city_id": self.city_id,
            "scenario_name": self.scenario_name,
            "decision_step": self.decision_step_count,
            "sim_time": self.adapter.get_current_time(),
            "episode_return": float(self.episode_return),
            "total_episode_return": float(self.total_episode_return),
            "intersection_ids": self.controlled_intersection_ids,
            "district_types": self._district_type_labels,
            "metrics": {
                "num_controlled_intersections": len(self.controlled_intersection_ids),
                "mean_reward": mean_reward,
                "mean_step_intersection_reward": self._mean_step_intersection_reward(),
                "mean_waiting_vehicles": float(avg_incoming_waiting.sum(axis=1).mean()),
                "mean_incoming_vehicles": float(avg_incoming_counts.sum(axis=1).mean()),
                "total_waiting_vehicles": float(avg_incoming_waiting.sum()),
                "total_incoming_vehicles": float(avg_incoming_counts.sum()),
                "running_vehicles": self.adapter.get_vehicle_count(),
                "throughput": self.adapter.get_finished_vehicle_count(),
                "average_travel_time": average_travel_time,
                "reward_variant": self.env_config.reward.variant,
            },
        }
        info["metrics"].update(self._reward_component_metrics(reward_components))
        info["metrics"].update(
            per_district_type_metrics(
                district_types=self._district_type_labels,
                rewards=rewards,
                avg_incoming_counts=avg_incoming_counts,
                avg_incoming_waiting=avg_incoming_waiting,
            )
        )
        return info

    def _normalize_actions(
        self,
        actions: dict[str, int] | list[int] | np.ndarray,
    ) -> np.ndarray:
        if isinstance(actions, dict):
            return np.asarray(
                [int(actions.get(intersection_id, 0)) for intersection_id in self.controlled_intersection_ids],
                dtype=np.int64,
            )
        array = np.asarray(actions, dtype=np.int64)
        if array.shape != (len(self.controlled_intersection_ids),):
            raise ValueError(
                "Actions must provide exactly one action per controlled intersection."
            )
        return array

    def _mean_step_intersection_reward(self) -> float:
        denominator = max(
            1,
            self.decision_step_count * len(self.controlled_intersection_ids),
        )
        return float(self.total_episode_return) / float(denominator)

    def _accumulate_reward_components(self, components: dict[str, np.ndarray]) -> None:
        for name, values in components.items():
            self.reward_component_sums[name] = self.reward_component_sums.get(name, 0.0) + float(
                np.asarray(values, dtype=np.float32).mean()
            )

    def _reward_component_metrics(
        self,
        reward_components: dict[str, np.ndarray],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, values in reward_components.items():
            metrics[f"reward_component_step_{name}"] = float(
                np.asarray(values, dtype=np.float32).mean()
            )
        if self.decision_step_count <= 0:
            return metrics
        for name, total in self.reward_component_sums.items():
            metrics[f"reward_component_mean_{name}"] = float(total) / float(
                self.decision_step_count
            )
        return metrics


def per_district_type_metrics(
    district_types: tuple[str, ...],
    rewards: np.ndarray,
    avg_incoming_counts: np.ndarray,
    avg_incoming_waiting: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    reward_vector = np.asarray(rewards, dtype=np.float32)
    incoming_totals = avg_incoming_counts.sum(axis=1)
    waiting_totals = avg_incoming_waiting.sum(axis=1)

    for district_type in sorted(set(district_types)):
        mask = np.asarray(
            [item == district_type for item in district_types],
            dtype=bool,
        )
        if not mask.any():
            continue
        metrics[f"num_{district_type}_intersections"] = float(mask.sum())
        metrics[f"mean_reward_{district_type}"] = float(reward_vector[mask].mean())
        metrics[f"mean_waiting_vehicles_{district_type}"] = float(waiting_totals[mask].mean())
        metrics[f"mean_incoming_vehicles_{district_type}"] = float(incoming_totals[mask].mean())

    return metrics
