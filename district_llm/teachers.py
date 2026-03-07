from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agents.local_policy import (
    BaseLocalPolicy,
    FixedCyclePolicy,
    HoldPhasePolicy,
    QueueGreedyPolicy,
    RandomPhasePolicy,
)


BASELINE_TYPES: tuple[str, ...] = ("hold", "fixed", "random", "queue_greedy")


@dataclass(frozen=True)
class TeacherMetadata:
    controller_type: str
    controller_id: str
    controller_family: str
    teacher_algorithm: str
    checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_type": self.controller_type,
            "controller_id": self.controller_id,
            "controller_family": self.controller_family,
            "teacher_algorithm": self.teacher_algorithm,
            "checkpoint_path": self.checkpoint_path,
        }


class BaseTeacher(ABC):
    def __init__(self, metadata: TeacherMetadata):
        self.metadata = metadata

    @property
    def env_config(self) -> Any | None:
        return None

    @abstractmethod
    def act(self, observation_batch: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class BaselineTeacher(BaseTeacher):
    def __init__(self, policy: BaseLocalPolicy, metadata: TeacherMetadata):
        super().__init__(metadata=metadata)
        self.policy = policy

    def act(self, observation_batch: dict[str, Any]) -> np.ndarray:
        return np.asarray(self.policy.act(observation_batch), dtype=np.int64)


class RLCheckpointTeacher(BaseTeacher):
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        deterministic: bool = True,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "RL checkpoint teachers require PyTorch to be installed."
            ) from exc

        from training.models import RunningNormalizer, TrafficControlQNetwork
        from training.train_local_policy import load_env_config

        checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        network_architecture = self.checkpoint.get("network_architecture") or self.checkpoint.get(
            "policy_architecture",
            {},
        )
        trainer_config = self.checkpoint.get("dqn_config", {})
        policy_arch = network_architecture.get(
            "policy_arch",
            trainer_config.get("policy_arch", "single_head_with_district_feature"),
        )
        self.model = TrafficControlQNetwork(
            observation_dim=int(network_architecture["observation_dim"]),
            action_dim=int(network_architecture.get("action_dim", 2)),
            hidden_dim=int(trainer_config.get("hidden_dim", 256)),
            num_layers=int(trainer_config.get("hidden_layers", 2)),
            district_types=tuple(network_architecture.get("district_types", ())),
            policy_arch=policy_arch,
            dueling=bool(network_architecture.get("dueling", True)),
        ).to(self.device)
        self.model.load_state_dict(
            self.checkpoint.get("q_network_state_dict") or self.checkpoint["policy_state_dict"]
        )
        self.model.eval()
        self.obs_normalizer = None
        if self.checkpoint.get("obs_normalizer"):
            self.obs_normalizer = RunningNormalizer()
            self.obs_normalizer.load_state_dict(self.checkpoint["obs_normalizer"])

        checkpoint_id = checkpoint_path.stem
        super().__init__(
            metadata=TeacherMetadata(
                controller_type="rl_checkpoint",
                controller_id=checkpoint_id,
                controller_family="dqn",
                teacher_algorithm="dqn",
                checkpoint_path=str(checkpoint_path),
            )
        )
        self.deterministic = bool(deterministic)
        self._env_config = (
            load_env_config(self.checkpoint["env_config"])
            if self.checkpoint.get("env_config")
            else None
        )

    @property
    def env_config(self) -> Any | None:
        return self._env_config

    def act(self, observation_batch: dict[str, Any]) -> np.ndarray:
        raw_obs = observation_batch["observations"].astype(np.float32)
        normalized_obs = self.obs_normalizer.normalize(raw_obs) if self.obs_normalizer else raw_obs
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
            actions = self.model.act(
                observations=obs_tensor,
                district_type_indices=district_type_tensor,
                action_mask=action_mask_tensor,
                deterministic=self.deterministic,
                epsilon=0.0,
            )
        return actions.cpu().numpy().astype(np.int64)


def build_teacher(
    controller_type: str,
    checkpoint: str | None = None,
    fixed_green_time: int = 20,
    seed: int = 7,
    device: str | None = None,
) -> BaseTeacher:
    if controller_type == "rl_checkpoint":
        if not checkpoint:
            raise ValueError("controller_type='rl_checkpoint' requires --checkpoint.")
        return RLCheckpointTeacher(checkpoint_path=checkpoint, device=device)
    if controller_type == "hold":
        return BaselineTeacher(
            policy=HoldPhasePolicy(),
            metadata=TeacherMetadata(
                controller_type="hold",
                controller_id="hold",
                controller_family="baseline",
                teacher_algorithm="hold",
            ),
        )
    if controller_type == "fixed":
        return BaselineTeacher(
            policy=FixedCyclePolicy(green_time=fixed_green_time),
            metadata=TeacherMetadata(
                controller_type="fixed",
                controller_id=f"fixed_{fixed_green_time}",
                controller_family="baseline",
                teacher_algorithm="fixed_cycle",
            ),
        )
    if controller_type == "random":
        return BaselineTeacher(
            policy=RandomPhasePolicy(seed=seed),
            metadata=TeacherMetadata(
                controller_type="random",
                controller_id=f"random_{seed}",
                controller_family="baseline",
                teacher_algorithm="random",
            ),
        )
    if controller_type == "queue_greedy":
        return BaselineTeacher(
            policy=QueueGreedyPolicy(),
            metadata=TeacherMetadata(
                controller_type="queue_greedy",
                controller_id="queue_greedy",
                controller_family="baseline",
                teacher_algorithm="queue_greedy",
            ),
        )
    raise ValueError(
        f"Unsupported controller_type '{controller_type}'. "
        f"Expected rl_checkpoint or one of {BASELINE_TYPES}."
    )


def parse_teacher_spec(spec: str) -> tuple[str, str | None]:
    if "=" not in spec:
        return spec.strip(), None
    controller_type, checkpoint_path = spec.split("=", 1)
    return controller_type.strip(), checkpoint_path.strip() or None


def teachers_metadata_json(teachers: list[BaseTeacher]) -> str:
    return json.dumps([teacher.metadata.to_dict() for teacher in teachers], sort_keys=True)
