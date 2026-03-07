from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from district_llm.guided_control import DistrictGuidedLocalController
from district_llm.schema import DistrictAction
from district_llm.summary_builder import DistrictStateSummaryBuilder
from district_llm.teachers import build_teacher, parse_teacher_spec
from env.observation_builder import ObservationConfig
from env.reward import RewardConfig
from env.traffic_env import EnvConfig, TrafficEnv
from training.dataset import CityFlowDataset


class OpenEnvTrafficWrapper:
    """
    OpenEnv-style district environment backed by the current DQN local stack.

    External action:
    - a dict of district-level directives keyed by district_id

    Internal execution:
    - the shared DQN (or a baseline fallback) produces low-level actions
    - district directives bias those low-level actions over a slower district window
    """

    def __init__(
        self,
        generated_root: str | Path = "data/generated",
        splits_root: str | Path = "data/splits",
        split: str = "train",
        controller_spec: str | None = None,
        district_decision_interval: int = 10,
        seed: int = 7,
    ):
        self.dataset = CityFlowDataset(
            generated_root=generated_root,
            splits_root=splits_root,
        )
        self.dataset.generate_default_splits()
        self.split = split
        self.rng = random.Random(seed)
        self.district_decision_interval = int(district_decision_interval)
        self.summary_builder = DistrictStateSummaryBuilder()

        default_checkpoint = Path("artifacts/dqn_shared/best_validation.pt")
        if controller_spec is None:
            controller_spec = (
                f"rl_checkpoint={default_checkpoint}"
                if default_checkpoint.exists()
                else "queue_greedy"
            )
        controller_type, checkpoint = parse_teacher_spec(controller_spec)
        try:
            self.teacher = build_teacher(
                controller_type=controller_type,
                checkpoint=checkpoint,
                seed=seed,
            )
        except ImportError:
            if controller_spec != "queue_greedy":
                self.teacher = build_teacher(
                    controller_type="queue_greedy",
                    checkpoint=None,
                    seed=seed,
                )
            else:
                raise
        self.guided_controller = DistrictGuidedLocalController(base_teacher=self.teacher)
        self.env_config = self.teacher.env_config or self._default_env_config()

        self.env: TrafficEnv | None = None
        self.current_scenario_spec = None
        self.last_obs: dict[str, Any] | None = None
        self.last_info: dict[str, Any] | None = None
        self.last_summaries: dict[str, Any] = {}

    def reset(
        self,
        seed: int | None = None,
        city_id: str | None = None,
        scenario_name: str | None = None,
    ) -> dict[str, Any]:
        scenario_spec = (
            self.dataset.build_scenario_spec(city_id, scenario_name)
            if city_id and scenario_name
            else self.dataset.sample_scenario(
                split_name=self.split,
                rng=self.rng,
                city_id=city_id,
                scenario_name=scenario_name,
            )
        )
        self.current_scenario_spec = scenario_spec
        self.env = TrafficEnv(
            city_id=scenario_spec.city_id,
            scenario_name=scenario_spec.scenario_name,
            city_dir=scenario_spec.city_dir,
            scenario_dir=scenario_spec.scenario_dir,
            config_path=scenario_spec.config_path,
            roadnet_path=scenario_spec.roadnet_path,
            district_map_path=scenario_spec.district_map_path,
            metadata_path=scenario_spec.metadata_path,
            env_config=self.env_config,
        )
        self.summary_builder.reset()
        self.last_obs = self.env.reset(seed=seed)
        self.last_summaries = self.summary_builder.build_all(self.env, self.last_obs)
        self.last_info = {
            "seed": seed,
            "city_id": scenario_spec.city_id,
            "scenario_name": scenario_spec.scenario_name,
            "controller_type": self.teacher.metadata.controller_type,
            "controller_family": self.teacher.metadata.controller_family,
            "teacher_algorithm": self.teacher.metadata.teacher_algorithm,
            "district_decision_interval": self.district_decision_interval,
        }
        return {
            "observation": self._build_observation_payload(),
            "info": self.last_info,
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.env is None or self.last_obs is None:
            self.reset(seed=None)

        assert self.env is not None
        district_actions = self._parse_district_actions(action.get("district_actions", {}))
        done = False
        reward_total = 0.0
        steps_executed = 0
        info: dict[str, Any] = {}

        for _ in range(self.district_decision_interval):
            local_actions = self.guided_controller.act(
                observation_batch=self.last_obs,
                district_actions=district_actions,
            )
            next_obs, rewards, done, info = self.env.step(local_actions)
            reward_total += float(np.asarray(rewards, dtype=np.float32).mean())
            self.last_obs = next_obs
            steps_executed += 1
            if done:
                break

        self.last_summaries = self.summary_builder.build_all(self.env, self.last_obs)
        self.last_info = {
            **info,
            "controller_type": self.teacher.metadata.controller_type,
            "controller_family": self.teacher.metadata.controller_family,
            "teacher_algorithm": self.teacher.metadata.teacher_algorithm,
            "steps_executed": steps_executed,
            "district_actions": {
                district_id: directive.to_dict()
                for district_id, directive in district_actions.items()
            },
        }
        return {
            "observation": self._build_observation_payload(),
            "reward": float(reward_total),
            "done": bool(done),
            "truncated": False,
            "info": self.last_info,
        }

    def state(self) -> dict[str, Any]:
        return {
            "state": {
                "scenario": (
                    None
                    if self.current_scenario_spec is None
                    else {
                        "city_id": self.current_scenario_spec.city_id,
                        "scenario_name": self.current_scenario_spec.scenario_name,
                    }
                ),
                "controller": self.teacher.metadata.to_dict(),
                "district_decision_interval": self.district_decision_interval,
                "district_summaries": {
                    district_id: summary.to_dict()
                    for district_id, summary in self.last_summaries.items()
                },
                "last_info": self.last_info or {},
            }
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "message": "DistrictFlow OpenEnv wrapper is running.",
        }

    def _build_observation_payload(self) -> dict[str, Any]:
        if self.env is None or self.last_obs is None:
            return {"district_summaries": {}}
        return {
            "city_id": self.env.city_id,
            "scenario_name": self.env.scenario_name,
            "decision_step": int(self.last_obs["decision_step"]),
            "sim_time": int(self.last_obs["sim_time"]),
            "district_summaries": {
                district_id: summary.to_dict()
                for district_id, summary in self.last_summaries.items()
            },
        }

    def _parse_district_actions(self, payload: dict[str, Any]) -> dict[str, DistrictAction]:
        if self.env is None:
            return {}
        parsed: dict[str, DistrictAction] = {}
        for district_id in self.env.districts:
            raw_action = payload.get(district_id)
            if raw_action is None:
                parsed[district_id] = DistrictAction.default_hold(
                    duration_steps=self.district_decision_interval
                )
                continue
            try:
                parsed[district_id] = DistrictAction.from_dict(raw_action)
            except Exception:
                parsed[district_id] = DistrictAction.default_hold(
                    duration_steps=self.district_decision_interval
                )
        return parsed

    @staticmethod
    def _default_env_config() -> EnvConfig:
        return EnvConfig(
            simulator_interval=1,
            decision_interval=5,
            min_green_time=10,
            thread_num=1,
            max_episode_seconds=None,
            observation=ObservationConfig(),
            reward=RewardConfig(variant="wait_queue_throughput"),
        )
