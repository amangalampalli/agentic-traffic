from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from itertools import islice
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from agents.local_policy import FixedCyclePolicy, HoldPhasePolicy, QueueGreedyPolicy, RandomPhasePolicy
from env.observation_builder import ObservationConfig
from env.reward import RewardConfig
from env.traffic_env import EnvConfig, TrafficEnv
from training.cityflow_dataset import CityFlowDataset, ScenarioSpec
from training.device import configure_torch_runtime, resolve_torch_device
from training.models import POLICY_ARCHES, RunningNormalizer, TrafficControlQNetwork
from training.rollout import evaluate_policy

_EVAL_CONTEXT: dict[str, Any] = {}


@dataclass(frozen=True)
class DQNConfig:
    policy_arch: str = "single_head_with_district_feature"
    total_updates: int = 200
    learning_rate: float = 1e-4
    gamma: float = 0.99
    n_step: int = 3
    replay_capacity: int = 500_000
    minibatch_size: int = 1024
    learning_starts: int = 10_000
    gradient_steps: int = 64
    target_tau: float = 0.01
    max_grad_norm: float = 10.0
    hidden_dim: int = 256
    hidden_layers: int = 2
    dueling: bool = True
    seed: int = 7
    eval_every: int = 40
    checkpoint_every: int = 40
    checkpoint_on_eval: bool = True
    val_scenarios_per_city: int | None = 1
    max_val_cities: int | None = 5
    max_train_cities: int | None = None
    num_rollout_workers: int = 4
    rollout_episodes_per_update: int | None = None
    train_city_id: str | None = None
    train_scenario_name: str | None = None
    overfit_val_on_train_scenario: bool = False
    rollout_decision_steps: int | None = 256
    resume_from: str | None = None
    use_observation_normalization: bool = True
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta_start: float = 0.4
    prioritized_replay_beta_end: float = 1.0
    prioritized_replay_beta_steps: int = 200_000
    compare_baselines: bool = True
    skip_failed_validation_episodes: bool = True
    verbose_progress: bool = False
    eval_num_workers: int = -1
    enable_tensorboard: bool = True
    tensorboard_log_dir: str | None = None
    rolling_window_size: int = 20
    use_tqdm: bool = True


@dataclass
class TrainerState:
    update_index: int = 0
    best_validation_score: float = float("-inf")
    total_decision_steps: int = 0
    total_transitions: int = 0
    gradient_steps: int = 0


@dataclass(frozen=True)
class StepRecord:
    observation: np.ndarray
    district_type_index: int
    action_mask: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    next_district_type_index: int
    next_action_mask: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        prioritized_alpha: float = 0.6,
        epsilon: float = 1e-6,
    ):
        self.capacity = int(capacity)
        self.prioritized_alpha = float(prioritized_alpha)
        self.epsilon = float(epsilon)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

        self.observations: np.ndarray | None = None
        self.next_observations: np.ndarray | None = None
        self.district_type_indices: np.ndarray | None = None
        self.next_district_type_indices: np.ndarray | None = None
        self.action_masks: np.ndarray | None = None
        self.next_action_masks: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.rewards: np.ndarray | None = None
        self.dones: np.ndarray | None = None
        self.discounts: np.ndarray | None = None
        self.priorities = np.zeros(self.capacity, dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        district_type_index: int,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_district_type_index: int,
        next_action_mask: np.ndarray,
        done: bool,
        discount: float,
    ) -> None:
        if self.observations is None:
            obs_dim = observation.shape[0]
            action_dim = action_mask.shape[0]
            self.observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
            self.next_observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
            self.district_type_indices = np.zeros(self.capacity, dtype=np.int64)
            self.next_district_type_indices = np.zeros(self.capacity, dtype=np.int64)
            self.action_masks = np.zeros((self.capacity, action_dim), dtype=np.float32)
            self.next_action_masks = np.zeros((self.capacity, action_dim), dtype=np.float32)
            self.actions = np.zeros(self.capacity, dtype=np.int64)
            self.rewards = np.zeros(self.capacity, dtype=np.float32)
            self.dones = np.zeros(self.capacity, dtype=np.float32)
            self.discounts = np.zeros(self.capacity, dtype=np.float32)

        index = self.position
        self.observations[index] = observation.astype(np.float32)
        self.next_observations[index] = next_observation.astype(np.float32)
        self.district_type_indices[index] = int(district_type_index)
        self.next_district_type_indices[index] = int(next_district_type_index)
        self.action_masks[index] = action_mask.astype(np.float32)
        self.next_action_masks[index] = next_action_mask.astype(np.float32)
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = float(done)
        self.discounts[index] = float(discount)
        self.priorities[index] = self.max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> dict[str, np.ndarray]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        replace = self.size < batch_size
        if self.prioritized_alpha > 0.0:
            scaled_priorities = np.power(
                np.maximum(self.priorities[: self.size], self.epsilon),
                self.prioritized_alpha,
            )
            probabilities = scaled_priorities / scaled_priorities.sum()
            indices = np.random.choice(
                self.size,
                size=batch_size,
                replace=replace,
                p=probabilities,
            )
            weights = np.power(self.size * probabilities[indices], -beta).astype(np.float32)
            weights /= max(1.0, float(weights.max()))
        else:
            indices = np.random.choice(self.size, size=batch_size, replace=replace)
            weights = np.ones(batch_size, dtype=np.float32)

        return {
            "indices": indices.astype(np.int64),
            "weights": weights.astype(np.float32),
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "district_type_indices": self.district_type_indices[indices],
            "next_district_type_indices": self.next_district_type_indices[indices],
            "action_masks": self.action_masks[indices],
            "next_action_masks": self.next_action_masks[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "discounts": self.discounts[indices],
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        updated_priorities = np.abs(td_errors).astype(np.float32) + self.epsilon
        self.priorities[indices] = updated_priorities
        if updated_priorities.size:
            self.max_priority = max(self.max_priority, float(updated_priorities.max()))

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "prioritized_alpha": self.prioritized_alpha,
            "epsilon": self.epsilon,
            "position": self.position,
            "size": self.size,
            "max_priority": self.max_priority,
            "observations": self.observations,
            "next_observations": self.next_observations,
            "district_type_indices": self.district_type_indices,
            "next_district_type_indices": self.next_district_type_indices,
            "action_masks": self.action_masks,
            "next_action_masks": self.next_action_masks,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "discounts": self.discounts,
            "priorities": self.priorities,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.capacity = int(state_dict["capacity"])
        self.prioritized_alpha = float(state_dict["prioritized_alpha"])
        self.epsilon = float(state_dict["epsilon"])
        self.position = int(state_dict["position"])
        self.size = int(state_dict["size"])
        self.max_priority = float(state_dict["max_priority"])
        self.observations = state_dict["observations"]
        self.next_observations = state_dict["next_observations"]
        self.district_type_indices = state_dict["district_type_indices"]
        self.next_district_type_indices = state_dict["next_district_type_indices"]
        self.action_masks = state_dict["action_masks"]
        self.next_action_masks = state_dict["next_action_masks"]
        self.actions = state_dict["actions"]
        self.rewards = state_dict["rewards"]
        self.dones = state_dict["dones"]
        self.discounts = state_dict["discounts"]
        self.priorities = state_dict["priorities"]


class DQNTrainer:
    def __init__(
        self,
        dataset: CityFlowDataset,
        env_config: EnvConfig,
        dqn_config: DQNConfig,
        output_dir: str | Path = "artifacts/dqn_shared",
        device: str | None = None,
    ):
        self.dataset = dataset
        self.env_config = env_config
        self.dqn_config = dqn_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.rng = random.Random(dqn_config.seed)
        np.random.seed(dqn_config.seed)
        torch.manual_seed(dqn_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(dqn_config.seed)

        self.device = resolve_torch_device(device)
        configure_torch_runtime(self.device)
        if self.dqn_config.policy_arch not in POLICY_ARCHES:
            raise ValueError(
                f"Unsupported policy architecture: {self.dqn_config.policy_arch}. "
                f"Expected one of {POLICY_ARCHES}."
            )

        self.train_city_ids = self.dataset.load_split("train")
        if self.dqn_config.max_train_cities is not None:
            self.train_city_ids = self.train_city_ids[: self.dqn_config.max_train_cities]
        self.fixed_train_scenario_spec = self._resolve_fixed_train_scenario()
        if not self.train_city_ids:
            raise ValueError("No training cities available for DQN training.")

        sample_spec = self._sample_train_scenario()
        sample_env = self._make_env(sample_spec)
        observation_dim = sample_env.observation_dim

        self.q_network = TrafficControlQNetwork(
            observation_dim=observation_dim,
            hidden_dim=dqn_config.hidden_dim,
            num_layers=dqn_config.hidden_layers,
            policy_arch=dqn_config.policy_arch,
            dueling=dqn_config.dueling,
        ).to(self.device)
        self.target_network = TrafficControlQNetwork(
            observation_dim=observation_dim,
            hidden_dim=dqn_config.hidden_dim,
            num_layers=dqn_config.hidden_layers,
            policy_arch=dqn_config.policy_arch,
            dueling=dqn_config.dueling,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=dqn_config.learning_rate)
        self.obs_normalizer = RunningNormalizer() if dqn_config.use_observation_normalization else None
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=dqn_config.replay_capacity,
            prioritized_alpha=dqn_config.prioritized_replay_alpha,
        )
        self.state = TrainerState()
        self.training_log_path = self.output_dir / "training_log.jsonl"
        self.validation_log_path = self.output_dir / "validation_log.jsonl"
        self.tensorboard_log_dir = Path(
            dqn_config.tensorboard_log_dir or (self.output_dir / "tensorboard")
        )
        self.writer = self._build_tensorboard_writer()
        self._rolling_metrics: dict[tuple[str, str], deque[float]] = {}
        self.rollout_executor: ProcessPoolExecutor | None = None
        if self.dqn_config.num_rollout_workers > 1:
            self.rollout_executor = ProcessPoolExecutor(
                max_workers=self.dqn_config.num_rollout_workers,
            )

        print(
            "[setup] "
            f"torch_device={self.device.type} "
            f"algorithm=ps_d3qn "
            f"policy_arch={self.dqn_config.policy_arch} "
            f"reward_variant={self.env_config.reward.variant} "
            f"rollout_workers={self.dqn_config.num_rollout_workers}"
        )
        if self.fixed_train_scenario_spec is not None:
            print(
                "[setup] "
                f"fixed_train_city={self.fixed_train_scenario_spec.city_id} "
                f"fixed_train_scenario={self.fixed_train_scenario_spec.scenario_name} "
                f"overfit_val_on_train_scenario={self.dqn_config.overfit_val_on_train_scenario}"
            )

        if dqn_config.resume_from:
            self.load_checkpoint(dqn_config.resume_from)

    def fit(self) -> None:
        progress_bar: tqdm | None = None
        try:
            if self.dqn_config.use_tqdm:
                progress_bar = tqdm(
                    total=self.dqn_config.total_updates,
                    initial=self.state.update_index,
                    desc="train",
                    dynamic_ncols=True,
                )
            for update_index in range(self.state.update_index, self.dqn_config.total_updates):
                rollout_start = perf_counter()
                episode_records = self._collect_rollout_batch()
                rollout_seconds = perf_counter() - rollout_start

                update_start = perf_counter()
                losses = self._optimize()
                update_seconds = perf_counter() - update_start

                self.state.update_index = update_index + 1
                validation_seconds = 0.0
                checkpoint_seconds = 0.0

                train_record = self._summarize_rollout_batch(episode_records)
                train_record.update(
                    {
                        "update": self.state.update_index,
                        "algorithm": "ps_d3qn",
                        "policy_arch": self.dqn_config.policy_arch,
                        "reward_variant": self.env_config.reward.variant,
                        "replay_size": float(self.replay_buffer.size),
                        "epsilon": float(self._epsilon()),
                        **losses,
                    }
                )
                self._attach_rolling_metrics(
                    namespace="train",
                    record=train_record,
                    keys=(
                        "episode_return",
                        "total_episode_return",
                        "mean_waiting_vehicles",
                        "throughput",
                        "td_loss",
                        "mean_q_value",
                        "mean_abs_td_error",
                    ),
                )
                self._append_jsonl(self.training_log_path, train_record)
                self._print_train_log(train_record)
                self._log_tensorboard_scalars("train", train_record, self.state.update_index)
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        ret=f"{train_record['episode_return']:.3f}",
                        wait=f"{train_record['mean_waiting_vehicles']:.2f}",
                        td=f"{train_record['td_loss']:.4f}",
                        eps=f"{train_record['epsilon']:.3f}",
                    )
                    progress_bar.update(1)

                should_evaluate = self.state.update_index % self.dqn_config.eval_every == 0
                should_periodic_checkpoint = (
                    self.state.update_index % self.dqn_config.checkpoint_every == 0
                )
                if should_periodic_checkpoint and not (
                    should_evaluate and self.dqn_config.checkpoint_on_eval
                ):
                    print(f"[train] saving checkpoint at update={self.state.update_index}")
                    checkpoint_start = perf_counter()
                    self.save_checkpoint(self.checkpoint_dir / f"update_{self.state.update_index:04d}.pt")
                    checkpoint_seconds += perf_counter() - checkpoint_start
                    print(f"[train] finished checkpoint at update={self.state.update_index}")

                if should_evaluate:
                    print(f"[train] starting validation at update={self.state.update_index}")
                    validation_start = perf_counter()
                    validation_record = self.evaluate_split("val")
                    validation_seconds = perf_counter() - validation_start
                    validation_record["update"] = self.state.update_index
                    validation_record["algorithm"] = "ps_d3qn"
                    validation_record["policy_arch"] = self.dqn_config.policy_arch
                    validation_record["reward_variant"] = self.env_config.reward.variant
                    self._attach_rolling_metrics(
                        namespace="eval",
                        record=validation_record,
                        keys=(
                            "mean_episode_return",
                            "mean_total_episode_return",
                            "mean_mean_waiting_vehicles",
                            "mean_throughput",
                        ),
                    )
                    self._append_jsonl(self.validation_log_path, validation_record)
                    self._print_eval_log(validation_record)
                    self._log_tensorboard_scalars("eval", validation_record, self.state.update_index)
                    print(f"[train] finished validation at update={self.state.update_index}")

                    if self.dqn_config.checkpoint_on_eval:
                        print(f"[train] saving checkpoint at update={self.state.update_index}")
                        checkpoint_start = perf_counter()
                        self.save_checkpoint(
                            self.checkpoint_dir / f"update_{self.state.update_index:04d}.pt"
                        )
                        checkpoint_seconds += perf_counter() - checkpoint_start
                        print(f"[train] finished checkpoint at update={self.state.update_index}")

                    validation_score = float(validation_record["mean_episode_return"])
                    if validation_score > self.state.best_validation_score:
                        self.state.best_validation_score = validation_score
                        print(f"[train] saving checkpoint at update={self.state.update_index}")
                        checkpoint_start = perf_counter()
                        self.save_checkpoint(self.output_dir / "best_validation.pt")
                        checkpoint_seconds += perf_counter() - checkpoint_start
                        print(f"[train] finished checkpoint at update={self.state.update_index}")

                print(
                    "[timing] "
                    f"rollout={rollout_seconds:.2f}s "
                    f"update={update_seconds:.2f}s "
                    f"validation={validation_seconds:.2f}s "
                    f"checkpoint={checkpoint_seconds:.2f}s"
                )

            print(f"[train] saving checkpoint at update={self.state.update_index}")
            final_checkpoint_start = perf_counter()
            self.save_checkpoint(self.output_dir / "last.pt")
            final_checkpoint_seconds = perf_counter() - final_checkpoint_start
            print(f"[train] finished checkpoint at update={self.state.update_index}")
            print(f"[timing] final_checkpoint={final_checkpoint_seconds:.2f}s")
        finally:
            if progress_bar is not None:
                progress_bar.close()
            if self.rollout_executor is not None:
                self.rollout_executor.shutdown(wait=True, cancel_futures=False)
            if self.writer is not None:
                self.writer.close()

    def evaluate_split(self, split_name: str) -> dict[str, float]:
        if split_name == "val" and self.dqn_config.overfit_val_on_train_scenario:
            if self.fixed_train_scenario_spec is None:
                raise ValueError(
                    "--overfit-val-on-train-scenario requires a fixed training city/scenario."
                )
            scenario_specs = [self.fixed_train_scenario_spec]
        else:
            scenario_specs = self.dataset.iter_scenarios(
                split_name=split_name,
                scenarios_per_city=self.dqn_config.val_scenarios_per_city,
                max_cities=self.dqn_config.max_val_cities,
                diversify_single_scenario=True,
            )
        if self._resolved_eval_workers(len(scenario_specs)) > 1:
            episode_metrics = self._evaluate_policy_parallel(scenario_specs)
        else:
            episode_metrics = self._evaluate_policy_sequential(scenario_specs)
        if not episode_metrics:
            raise RuntimeError("Validation produced no successful episodes.")
        aggregate = aggregate_metrics(episode_metrics)
        aggregate.update(aggregate_metrics_by_scenario(episode_metrics))
        if self.dqn_config.compare_baselines:
            if self._resolved_eval_workers(len(scenario_specs)) > 1:
                aggregate.update(self._evaluate_baselines_parallel(scenario_specs))
            else:
                aggregate.update(self._evaluate_baselines(scenario_specs))
            if "fixed_mean_episode_return" in aggregate:
                aggregate["learner_minus_fixed_return"] = (
                    aggregate["mean_episode_return"] - aggregate["fixed_mean_episode_return"]
                )
            if "random_mean_episode_return" in aggregate:
                aggregate["learner_minus_random_return"] = (
                    aggregate["mean_episode_return"] - aggregate["random_mean_episode_return"]
                )
        return aggregate

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint = {
            "algorithm": "ps_d3qn",
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_state": asdict(self.state),
            "dqn_config": asdict(self.dqn_config),
            "network_architecture": {
                "observation_dim": self.q_network.observation_dim,
                "action_dim": self.q_network.action_dim,
                "district_types": self.q_network.district_types,
                "policy_arch": self.q_network.policy_arch,
                "dueling": self.q_network.dueling,
            },
            "env_config": {
                "simulator_interval": self.env_config.simulator_interval,
                "decision_interval": self.env_config.decision_interval,
                "min_green_time": self.env_config.min_green_time,
                "thread_num": self.env_config.thread_num,
                "max_episode_seconds": self.env_config.max_episode_seconds,
                "observation": asdict(self.env_config.observation),
                "reward": asdict(self.env_config.reward),
            },
            "obs_normalizer": self.obs_normalizer.state_dict() if self.obs_normalizer else None,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        q_state_dict = checkpoint.get("q_network_state_dict") or checkpoint.get("policy_state_dict")
        if q_state_dict is None:
            raise ValueError(f"Checkpoint at {path} does not contain a Q-network state dict.")
        self.q_network.load_state_dict(q_state_dict)
        target_state_dict = checkpoint.get("target_network_state_dict") or q_state_dict
        self.target_network.load_state_dict(target_state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state = TrainerState(**checkpoint["trainer_state"])
        if self.obs_normalizer and checkpoint.get("obs_normalizer"):
            self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

    def _make_env(self, scenario_spec: ScenarioSpec) -> TrafficEnv:
        return TrafficEnv(
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

    def _collect_rollout_batch(self) -> list[dict[str, float | str]]:
        episodes_per_update = self.dqn_config.rollout_episodes_per_update or max(
            1,
            self.dqn_config.num_rollout_workers,
        )
        scenario_specs = [self._sample_train_scenario() for _ in range(episodes_per_update)]
        if self.rollout_executor is None or episodes_per_update <= 1:
            episode_record = self._collect_episode(self._make_env(scenario_specs[0]))
            return [episode_record]
        return self._collect_rollouts_parallel(scenario_specs)

    def _sample_train_scenario(self) -> ScenarioSpec:
        if self.fixed_train_scenario_spec is not None:
            return self.fixed_train_scenario_spec
        selected_city = self.rng.choice(self.train_city_ids)
        selected_scenario = self.rng.choice(self.dataset.scenarios_for_city(selected_city))
        if self.dqn_config.verbose_progress:
            print(f"[train] sampled city={selected_city} scenario={selected_scenario}")
        return self.dataset.build_scenario_spec(selected_city, selected_scenario)

    def _resolve_fixed_train_scenario(self) -> ScenarioSpec | None:
        if self.dqn_config.train_city_id is None:
            return None

        city_id = self.dqn_config.train_city_id
        available_train_cities = set(self.dataset.load_split("train"))
        if city_id not in available_train_cities:
            raise ValueError(
                f"Fixed train city {city_id!r} is not in the train split."
            )

        scenario_names = self.dataset.scenarios_for_city(city_id)
        scenario_name = self.dqn_config.train_scenario_name
        if scenario_name is None:
            scenario_name = scenario_names[0]
        if scenario_name not in scenario_names:
            raise ValueError(
                f"Scenario {scenario_name!r} not found for train city {city_id!r}. "
                f"Available: {scenario_names}"
            )

        self.train_city_ids = [city_id]
        return self.dataset.build_scenario_spec(city_id, scenario_name)

    def _collect_rollouts_parallel(
        self,
        scenario_specs: list[ScenarioSpec],
    ) -> list[dict[str, float | str]]:
        if self.rollout_executor is None:
            raise RuntimeError("Parallel rollout collection requested without a rollout executor.")

        context = self._build_parallel_rollout_context()
        epsilon = self._epsilon()
        total_specs = len(scenario_specs)
        episode_records: list[dict[str, float | str]] = []
        futures = {
            self.rollout_executor.submit(
                _parallel_rollout_collection_worker,
                spec,
                context,
                epsilon,
                self.dqn_config.rollout_decision_steps,
                self.dqn_config.gamma,
                self.dqn_config.n_step,
            ): (index, spec)
            for index, spec in enumerate(scenario_specs, start=1)
        }
        for future in as_completed(futures):
            index, spec = futures[future]
            result = future.result()
            self._ingest_transition_batch(result["transitions"])
            self.state.total_decision_steps += int(result["episode_record"]["decision_steps"])
            self.state.total_transitions += int(result["episode_record"]["transitions"])
            episode_records.append(result["episode_record"])
            if self.dqn_config.verbose_progress:
                print(
                    f"[rollout] city={spec.city_id} scenario={spec.scenario_name} "
                    f"i={index}/{total_specs}"
                )
        return episode_records

    def _build_parallel_rollout_context(self) -> dict[str, Any]:
        return {
            "env_config": _env_config_to_payload(self.env_config),
            "network_architecture": {
                "observation_dim": self.q_network.observation_dim,
                "action_dim": self.q_network.action_dim,
                "hidden_dim": self.q_network.hidden_dim,
                "num_layers": self.q_network.num_layers,
                "district_types": self.q_network.district_types,
                "policy_arch": self.q_network.policy_arch,
                "dueling": self.q_network.dueling,
            },
            "q_network_state_dict": {
                key: value.detach().cpu()
                for key, value in self.q_network.state_dict().items()
            },
            "obs_normalizer": self.obs_normalizer.state_dict() if self.obs_normalizer else None,
        }

    def _ingest_transition_batch(self, transitions: dict[str, np.ndarray]) -> None:
        if transitions["observations"].size == 0:
            return
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(transitions["observations"])
        transition_count = transitions["actions"].shape[0]
        for index in range(transition_count):
            self.replay_buffer.add(
                observation=transitions["observations"][index],
                district_type_index=int(transitions["district_type_indices"][index]),
                action_mask=transitions["action_masks"][index],
                action=int(transitions["actions"][index]),
                reward=float(transitions["rewards"][index]),
                next_observation=transitions["next_observations"][index],
                next_district_type_index=int(transitions["next_district_type_indices"][index]),
                next_action_mask=transitions["next_action_masks"][index],
                done=bool(transitions["dones"][index]),
                discount=float(transitions["discounts"][index]),
            )

    def _summarize_rollout_batch(
        self,
        episode_records: list[dict[str, float | str]],
    ) -> dict[str, float | str]:
        if len(episode_records) == 1:
            record = dict(episode_records[0])
            record["num_rollout_episodes"] = 1.0
            return record

        aggregate = aggregate_metrics(episode_records)
        city_ids = sorted({str(record["city_id"]) for record in episode_records})
        scenario_names = sorted({str(record["scenario_name"]) for record in episode_records})
        summary: dict[str, float | str] = {
            "city_id": city_ids[0] if len(city_ids) == 1 else f"{len(city_ids)}_cities",
            "scenario_name": scenario_names[0]
            if len(scenario_names) == 1
            else f"{len(scenario_names)}_scenarios",
            "num_rollout_episodes": float(len(episode_records)),
        }
        for source_key, target_key in (
            ("mean_decision_steps", "decision_steps"),
            ("mean_transitions", "transitions"),
            ("mean_episode_return", "episode_return"),
            ("mean_total_episode_return", "total_episode_return"),
            ("mean_mean_waiting_vehicles", "mean_waiting_vehicles"),
            ("mean_throughput", "throughput"),
            ("mean_mean_q_value", "mean_q_value"),
            ("mean_epsilon", "epsilon"),
            ("mean_replay_size", "replay_size"),
        ):
            if source_key in aggregate:
                summary[target_key] = aggregate[source_key]
        for key, value in aggregate.items():
            if key not in summary:
                summary[key] = value
        return summary

    def _collect_episode(self, env: TrafficEnv) -> dict[str, float | str]:
        observation_batch = env.reset()
        decision_steps = 0
        transitions_added = 0
        q_value_samples: list[float] = []
        n_step_buffers = [
            deque() for _ in range(len(observation_batch["intersection_ids"]))
        ]
        epsilon = self._epsilon()
        last_info = env.last_info

        done = False
        while not done:
            if (
                self.dqn_config.rollout_decision_steps is not None
                and decision_steps >= self.dqn_config.rollout_decision_steps
            ):
                break

            raw_obs = observation_batch["observations"].astype(np.float32)
            if self.obs_normalizer is not None:
                self.obs_normalizer.update(raw_obs)
                normalized_obs = self.obs_normalizer.normalize(raw_obs)
            else:
                normalized_obs = raw_obs

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
                q_values = self.q_network.forward(
                    observations=obs_tensor,
                    district_type_indices=district_type_tensor,
                    action_mask=action_mask_tensor,
                )
                action_tensor = self.q_network.act(
                    observations=obs_tensor,
                    district_type_indices=district_type_tensor,
                    action_mask=action_mask_tensor,
                    deterministic=False,
                    epsilon=epsilon,
                )
            q_value_samples.append(float(q_values.max(dim=-1).values.mean().detach().cpu()))
            actions = action_tensor.detach().cpu().numpy()

            next_observation_batch, rewards, done, info = env.step(actions)
            transitions_added += self._append_step_records(
                buffers=n_step_buffers,
                observation_batch=observation_batch,
                actions=actions,
                rewards=np.asarray(rewards, dtype=np.float32),
                next_observation_batch=next_observation_batch,
                done=done,
            )
            observation_batch = next_observation_batch
            last_info = info
            decision_steps += 1
            self.state.total_decision_steps += 1
            epsilon = self._epsilon()

        transitions_added += self._flush_n_step_buffers(n_step_buffers)
        self.state.total_transitions += transitions_added

        episode_metrics = {
            key: float(value)
            for key, value in last_info["metrics"].items()
            if value is not None and isinstance(value, (int, float))
        }
        episode_metrics.update(
            {
                "city_id": env.city_id,
                "scenario_name": env.scenario_name,
                "decision_steps": decision_steps,
                "transitions": transitions_added,
                "episode_return": float(env.episode_return),
                "total_episode_return": float(env.total_episode_return),
                "epsilon": float(epsilon),
                "replay_size": float(self.replay_buffer.size),
                "mean_q_value": float(np.mean(q_value_samples)) if q_value_samples else 0.0,
            }
        )
        return episode_metrics

    def _append_step_records(
        self,
        buffers: list[deque[StepRecord]],
        observation_batch: dict[str, Any],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observation_batch: dict[str, Any],
        done: bool,
    ) -> int:
        transitions_added = 0
        for row_index, buffer in enumerate(buffers):
            record = StepRecord(
                observation=observation_batch["observations"][row_index].astype(np.float32),
                district_type_index=int(observation_batch["district_type_indices"][row_index]),
                action_mask=observation_batch["action_mask"][row_index].astype(np.float32),
                action=int(actions[row_index]),
                reward=float(rewards[row_index]),
                next_observation=next_observation_batch["observations"][row_index].astype(np.float32),
                next_district_type_index=int(next_observation_batch["district_type_indices"][row_index]),
                next_action_mask=next_observation_batch["action_mask"][row_index].astype(np.float32),
                done=bool(done),
            )
            buffer.append(record)
            if len(buffer) >= self.dqn_config.n_step:
                self._push_n_step_transition(buffer, steps=self.dqn_config.n_step)
                transitions_added += 1
        return transitions_added

    def _flush_n_step_buffers(self, buffers: list[deque[StepRecord]]) -> int:
        transitions_added = 0
        for buffer in buffers:
            while buffer:
                self._push_n_step_transition(buffer, steps=len(buffer))
                transitions_added += 1
        return transitions_added

    def _push_n_step_transition(self, buffer: deque[StepRecord], steps: int) -> None:
        records = list(islice(buffer, 0, steps))
        reward = 0.0
        for step_index, record in enumerate(records):
            reward += (self.dqn_config.gamma ** step_index) * float(record.reward)

        first_record = records[0]
        last_record = records[-1]
        discount = self.dqn_config.gamma ** len(records)
        self.replay_buffer.add(
            observation=first_record.observation,
            district_type_index=first_record.district_type_index,
            action_mask=first_record.action_mask,
            action=first_record.action,
            reward=reward,
            next_observation=last_record.next_observation,
            next_district_type_index=last_record.next_district_type_index,
            next_action_mask=last_record.next_action_mask,
            done=last_record.done,
            discount=discount,
        )
        buffer.popleft()

    def _optimize(self) -> dict[str, float]:
        minimum_replay = max(self.dqn_config.learning_starts, self.dqn_config.minibatch_size)
        if self.replay_buffer.size < minimum_replay:
            return {
                "td_loss": 0.0,
                "mean_abs_td_error": 0.0,
                "mean_target_q": 0.0,
                "mean_q_value": 0.0,
                "beta": self._beta(),
                "gradient_steps": 0.0,
            }

        batch_size = min(self.dqn_config.minibatch_size, self.replay_buffer.size)
        td_losses: list[float] = []
        td_errors: list[float] = []
        target_values: list[float] = []
        q_values: list[float] = []
        beta = self._beta()

        for _ in range(self.dqn_config.gradient_steps):
            batch = self.replay_buffer.sample(batch_size=batch_size, beta=beta)
            observations = batch["observations"]
            next_observations = batch["next_observations"]
            if self.obs_normalizer is not None:
                observations = self.obs_normalizer.normalize(observations)
                next_observations = self.obs_normalizer.normalize(next_observations)

            obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            next_obs_tensor = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
            district_type_tensor = torch.as_tensor(
                batch["district_type_indices"],
                dtype=torch.int64,
                device=self.device,
            )
            next_district_type_tensor = torch.as_tensor(
                batch["next_district_type_indices"],
                dtype=torch.int64,
                device=self.device,
            )
            action_mask_tensor = torch.as_tensor(batch["action_masks"], dtype=torch.float32, device=self.device)
            next_action_mask_tensor = torch.as_tensor(
                batch["next_action_masks"],
                dtype=torch.float32,
                device=self.device,
            )
            action_tensor = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
            reward_tensor = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
            done_tensor = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
            discount_tensor = torch.as_tensor(batch["discounts"], dtype=torch.float32, device=self.device)
            weight_tensor = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

            predicted_q = self.q_network.q_values_for_actions(
                observations=obs_tensor,
                district_type_indices=district_type_tensor,
                actions=action_tensor,
                action_mask=action_mask_tensor,
            )

            with torch.no_grad():
                next_online_q = self.q_network.forward(
                    observations=next_obs_tensor,
                    district_type_indices=next_district_type_tensor,
                    action_mask=next_action_mask_tensor,
                )
                next_actions = next_online_q.argmax(dim=-1)
                next_target_q = self.target_network.forward(
                    observations=next_obs_tensor,
                    district_type_indices=next_district_type_tensor,
                    action_mask=next_action_mask_tensor,
                ).gather(dim=1, index=next_actions.view(-1, 1)).squeeze(1)
                target_q = reward_tensor + (1.0 - done_tensor) * discount_tensor * next_target_q

            td_error = target_q - predicted_q
            per_sample_loss = nn.functional.smooth_l1_loss(
                predicted_q,
                target_q,
                reduction="none",
            )
            loss = (weight_tensor * per_sample_loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.dqn_config.max_grad_norm)
            self.optimizer.step()

            self._soft_update_target()
            self.replay_buffer.update_priorities(
                batch["indices"],
                td_errors=np.abs(td_error.detach().cpu().numpy()),
            )
            self.state.gradient_steps += 1

            td_losses.append(float(loss.detach().cpu()))
            td_errors.append(float(torch.abs(td_error).mean().detach().cpu()))
            target_values.append(float(target_q.mean().detach().cpu()))
            q_values.append(float(predicted_q.mean().detach().cpu()))

        return {
            "td_loss": float(np.mean(td_losses)),
            "mean_abs_td_error": float(np.mean(td_errors)),
            "mean_target_q": float(np.mean(target_values)),
            "mean_q_value": float(np.mean(q_values)),
            "beta": float(beta),
            "gradient_steps": float(self.dqn_config.gradient_steps),
        }

    def _soft_update_target(self) -> None:
        tau = float(self.dqn_config.target_tau)
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_network.parameters(),
                self.q_network.parameters(),
                strict=True,
            ):
                target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

    def _epsilon(self) -> float:
        if self.dqn_config.epsilon_decay_steps <= 0:
            return float(self.dqn_config.epsilon_end)
        progress = min(1.0, self.state.total_decision_steps / float(self.dqn_config.epsilon_decay_steps))
        return float(
            self.dqn_config.epsilon_start
            + progress * (self.dqn_config.epsilon_end - self.dqn_config.epsilon_start)
        )

    def _beta(self) -> float:
        if self.dqn_config.prioritized_replay_beta_steps <= 0:
            return float(self.dqn_config.prioritized_replay_beta_end)
        progress = min(
            1.0,
            self.state.total_decision_steps / float(self.dqn_config.prioritized_replay_beta_steps),
        )
        return float(
            self.dqn_config.prioritized_replay_beta_start
            + progress
            * (
                self.dqn_config.prioritized_replay_beta_end
                - self.dqn_config.prioritized_replay_beta_start
            )
        )

    def _evaluate_policy_sequential(
        self,
        scenario_specs: list[ScenarioSpec],
    ) -> list[dict[str, float | str]]:
        episode_metrics: list[dict[str, float | str]] = []
        total_specs = len(scenario_specs)
        iterator = enumerate(scenario_specs, start=1)
        if self.dqn_config.use_tqdm:
            iterator = tqdm(
                iterator,
                total=total_specs,
                desc="eval:learned",
                leave=False,
                dynamic_ncols=True,
            )
        for index, spec in iterator:
            print(f"[eval] city={spec.city_id} scenario={spec.scenario_name} i={index}/{total_specs}")
            try:
                episode_metrics.append(
                    evaluate_policy(
                        env_factory=lambda spec=spec: self._make_env(spec),
                        actor=self.q_network,
                        device=self.device,
                        obs_normalizer=self.obs_normalizer,
                        deterministic=True,
                    )
                )
            except Exception as exc:
                self._handle_eval_failure("validation", spec, exc)
        return episode_metrics

    def _evaluate_policy_parallel(
        self,
        scenario_specs: list[ScenarioSpec],
    ) -> list[dict[str, float | str]]:
        resolved_workers = self._resolved_eval_workers(len(scenario_specs))
        print(f"[eval] learned_workers={resolved_workers}")
        return self._run_parallel_eval(
            scenario_specs=scenario_specs,
            worker_kind="learned",
            initializer=_init_parallel_learned_eval_worker,
            initargs=(self._build_parallel_learned_eval_context(),),
            max_workers=resolved_workers,
        )

    def _append_jsonl(self, path: Path, record: dict) -> None:
        with path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def _build_tensorboard_writer(self) -> SummaryWriter | None:
        if not self.dqn_config.enable_tensorboard:
            return None
        if SummaryWriter is None:
            print("[setup] tensorboard_disabled=torch.utils.tensorboard unavailable")
            return None
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(self.tensorboard_log_dir))

    def _log_tensorboard_scalars(
        self,
        namespace: str,
        record: dict[str, Any],
        step: int,
    ) -> None:
        if self.writer is None:
            return
        for key, value in record.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{namespace}/{key}", float(value), step)
        self.writer.flush()

    def _attach_rolling_metrics(
        self,
        namespace: str,
        record: dict[str, Any],
        keys: tuple[str, ...],
    ) -> None:
        for key in keys:
            value = record.get(key)
            if not isinstance(value, (int, float)):
                continue
            window = self._rolling_metrics.setdefault(
                (namespace, key),
                deque(maxlen=self.dqn_config.rolling_window_size),
            )
            window.append(float(value))
            record[f"rolling_{key}"] = float(np.mean(window))

    def _evaluate_baselines(self, scenario_specs: list[ScenarioSpec]) -> dict[str, float]:
        baseline_metrics: dict[str, float] = {}
        for baseline_name in ("random", "fixed"):
            metrics: list[dict[str, float | str]] = []
            total_specs = len(scenario_specs)
            for offset, spec in enumerate(scenario_specs, start=1):
                print(
                    f"[eval] baseline={baseline_name} city={spec.city_id} "
                    f"scenario={spec.scenario_name} i={offset}/{total_specs}"
                )
                try:
                    actor = (
                        RandomPhasePolicy(seed=self.dqn_config.seed + offset)
                        if baseline_name == "random"
                        else FixedCyclePolicy(green_time=max(20, self.env_config.min_green_time * 2))
                    )
                    metrics.append(
                        evaluate_policy(
                            env_factory=lambda spec=spec: self._make_env(spec),
                            actor=actor,
                        )
                    )
                except Exception as exc:
                    message = (
                        f"[warn] baseline={baseline_name} failed for city={spec.city_id} "
                        f"scenario={spec.scenario_name}: {exc}"
                    )
                    if self.dqn_config.skip_failed_validation_episodes:
                        print(message)
                        continue
                    raise RuntimeError(message) from exc
            if not metrics:
                continue
            aggregate = aggregate_metrics(metrics)
            for key, value in aggregate.items():
                baseline_metrics[f"{baseline_name}_{key}"] = value
        return baseline_metrics

    def _evaluate_baselines_parallel(self, scenario_specs: list[ScenarioSpec]) -> dict[str, float]:
        baseline_metrics: dict[str, float] = {}
        resolved_workers = self._resolved_eval_workers(len(scenario_specs))
        print(f"[eval] baseline_workers={resolved_workers}")
        for baseline_name in ("random", "fixed"):
            metrics = self._run_parallel_eval(
                scenario_specs=scenario_specs,
                worker_kind=baseline_name,
                initializer=_init_parallel_baseline_worker,
                initargs=(self._build_parallel_baseline_context(baseline_name),),
                max_workers=resolved_workers,
            )
            if not metrics:
                continue
            aggregate = aggregate_metrics(metrics)
            for key, value in aggregate.items():
                baseline_metrics[f"{baseline_name}_{key}"] = value
        return baseline_metrics

    def _run_parallel_eval(
        self,
        scenario_specs: list[ScenarioSpec],
        worker_kind: str,
        initializer,
        initargs: tuple[Any, ...],
        max_workers: int,
    ) -> list[dict[str, float | str]]:
        metrics: list[dict[str, float | str]] = []
        total_specs = len(scenario_specs)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initializer,
            initargs=initargs,
        ) as executor:
            futures = {
                executor.submit(_parallel_eval_worker, spec, index, worker_kind): (spec, index)
                for index, spec in enumerate(scenario_specs, start=1)
            }
            iterator = as_completed(futures)
            if self.dqn_config.use_tqdm:
                iterator = tqdm(
                    iterator,
                    total=total_specs,
                    desc=f"eval:{worker_kind}",
                    leave=False,
                    dynamic_ncols=True,
                )
            for future in iterator:
                spec, index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self._handle_eval_failure(worker_kind, spec, exc)
                    continue
                prefix = f"[eval] baseline={worker_kind}"
                print(f"{prefix} city={spec.city_id} scenario={spec.scenario_name} i={index}/{total_specs}")
                metrics.append(result)
        return metrics

    def _handle_eval_failure(
        self,
        phase: str,
        spec: ScenarioSpec,
        exc: Exception,
    ) -> None:
        message = f"[warn] {phase} failed for city={spec.city_id} scenario={spec.scenario_name}: {exc}"
        if self.dqn_config.skip_failed_validation_episodes:
            print(message)
            return
        raise RuntimeError(message) from exc

    def _build_parallel_baseline_context(self, baseline_name: str) -> dict[str, Any]:
        return {
            "env_config": _env_config_to_payload(self.env_config),
            "baseline_name": baseline_name,
            "fixed_green_time": max(20, self.env_config.min_green_time * 2),
            "seed": self.dqn_config.seed,
        }

    def _build_parallel_learned_eval_context(self) -> dict[str, Any]:
        return {
            "env_config": _env_config_to_payload(self.env_config),
            "network_architecture": {
                "observation_dim": self.q_network.observation_dim,
                "action_dim": self.q_network.action_dim,
                "hidden_dim": self.q_network.hidden_dim,
                "num_layers": self.q_network.num_layers,
                "district_types": self.q_network.district_types,
                "policy_arch": self.q_network.policy_arch,
                "dueling": self.q_network.dueling,
            },
            "q_network_state_dict": {
                key: value.detach().cpu()
                for key, value in self.q_network.state_dict().items()
            },
            "obs_normalizer": self.obs_normalizer.state_dict() if self.obs_normalizer else None,
        }

    def _resolved_eval_workers(self, total_specs: int) -> int:
        requested = self.dqn_config.eval_num_workers
        if requested == -1:
            requested = os.cpu_count() or 1
        if requested <= 1:
            return 1
        return min(requested, total_specs)

    def _print_train_log(self, record: dict[str, float | str]) -> None:
        message = (
            "[train] "
            f"update={record['update']} algo={record['algorithm']} arch={record['policy_arch']} "
            f"reward={record['reward_variant']} episodes={int(record.get('num_rollout_episodes', 1.0))} "
            f"city={record['city_id']} scenario={record['scenario_name']} "
            f"mean_return={record['episode_return']:.3f} "
            f"(avg={record.get('rolling_episode_return', record['episode_return']):.3f}) "
            f"wait={record['mean_waiting_vehicles']:.3f} "
            f"(avg={record.get('rolling_mean_waiting_vehicles', record['mean_waiting_vehicles']):.3f}) "
            f"throughput={record['throughput']:.1f} "
            f"(avg={record.get('rolling_throughput', record['throughput']):.1f}) "
            f"epsilon={record['epsilon']:.3f} replay={int(record['replay_size'])} "
            f"td_loss={record['td_loss']:.4f} "
            f"(avg={record.get('rolling_td_loss', record['td_loss']):.4f}) "
            f"q={record['mean_q_value']:.4f} "
            f"td_err={record['mean_abs_td_error']:.4f}"
        )
        if self.dqn_config.use_tqdm:
            tqdm.write(message)
        else:
            print(message)

    def _print_eval_log(self, record: dict[str, float]) -> None:
        message = (
            "[eval] "
            f"algo={record['algorithm']} arch={record['policy_arch']} reward={record['reward_variant']} "
            f"episodes={int(record['num_episodes'])} "
            f"mean_return={record['mean_episode_return']:.3f} "
            f"(avg={record.get('rolling_mean_episode_return', record['mean_episode_return']):.3f}) "
            f"wait={record['mean_mean_waiting_vehicles']:.3f} "
            f"throughput={record['mean_throughput']:.1f} "
            f"travel_time={record.get('mean_average_travel_time', float('nan')):.3f}"
        )
        if self.dqn_config.compare_baselines:
            message += (
                f" fixed={record.get('fixed_mean_episode_return', float('nan')):.3f}"
                f" random={record.get('random_mean_episode_return', float('nan')):.3f}"
                f" vs_fixed={record.get('learner_minus_fixed_return', float('nan')):.3f}"
                f" vs_random={record.get('learner_minus_random_return', float('nan')):.3f}"
            )
        if self.dqn_config.use_tqdm:
            tqdm.write(message)
        else:
            print(message)
        scenario_summaries = []
        for scenario_name in (
            "accident",
            "construction",
            "district_overload",
            "evening_rush",
            "event_spike",
            "morning_rush",
            "normal",
        ):
            key = f"scenario_{scenario_name}_mean_episode_return"
            if key in record:
                scenario_summaries.append(f"{scenario_name}={record[key]:.3f}")
        if scenario_summaries:
            if self.dqn_config.use_tqdm:
                tqdm.write("[eval] scenario_returns " + " ".join(scenario_summaries))
            else:
                print("[eval] scenario_returns " + " ".join(scenario_summaries))


def aggregate_metrics(metrics: list[dict[str, float | str]]) -> dict[str, float]:
    numeric_keys = {
        key
        for item in metrics
        for key, value in item.items()
        if isinstance(value, (int, float))
    }
    aggregate = {"num_episodes": float(len(metrics))}
    for key in sorted(numeric_keys):
        aggregate[f"mean_{key}"] = float(
            np.mean([float(item[key]) for item in metrics if key in item])
        )
    return aggregate


def aggregate_metrics_by_scenario(metrics: list[dict[str, float | str]]) -> dict[str, float]:
    scenario_names = sorted(
        {
            str(item["scenario_name"])
            for item in metrics
            if isinstance(item.get("scenario_name"), str)
        }
    )
    aggregate: dict[str, float] = {}
    for scenario_name in scenario_names:
        scenario_metrics = [item for item in metrics if item.get("scenario_name") == scenario_name]
        if not scenario_metrics:
            continue
        scenario_aggregate = aggregate_metrics(scenario_metrics)
        for key, value in scenario_aggregate.items():
            aggregate[f"scenario_{scenario_name}_{key}"] = value
    return aggregate


def _env_config_to_payload(env_config: EnvConfig) -> dict[str, Any]:
    return {
        "simulator_interval": env_config.simulator_interval,
        "decision_interval": env_config.decision_interval,
        "min_green_time": env_config.min_green_time,
        "thread_num": env_config.thread_num,
        "max_episode_seconds": env_config.max_episode_seconds,
        "observation": asdict(env_config.observation),
        "reward": asdict(env_config.reward),
    }


def _env_config_from_payload(payload: dict[str, Any]) -> EnvConfig:
    return EnvConfig(
        simulator_interval=payload["simulator_interval"],
        decision_interval=payload["decision_interval"],
        min_green_time=payload["min_green_time"],
        thread_num=payload["thread_num"],
        max_episode_seconds=payload["max_episode_seconds"],
        observation=ObservationConfig(**payload["observation"]),
        reward=RewardConfig(**payload["reward"]),
    )


def _init_parallel_baseline_worker(context: dict[str, Any]) -> None:
    _init_parallel_eval_worker_from_context(context)


def _init_parallel_learned_eval_worker(context: dict[str, Any]) -> None:
    _init_parallel_eval_worker_from_context(context)


def _build_standalone_eval_context(
    env_config: EnvConfig,
    actor: TrafficControlQNetwork | RandomPhasePolicy | FixedCyclePolicy | HoldPhasePolicy | QueueGreedyPolicy,
    obs_normalizer: RunningNormalizer | None,
    device: torch.device,
    seed: int,
    fixed_green_time: int,
    baseline_name: str | None,
) -> dict[str, Any]:
    del device
    if baseline_name is not None:
        return {
            "env_config": _env_config_to_payload(env_config),
            "baseline_name": baseline_name,
            "fixed_green_time": fixed_green_time,
            "seed": seed,
        }

    if not isinstance(actor, TrafficControlQNetwork):
        raise ValueError("Standalone parallel learned evaluation requires a Q-network actor.")
    return {
        "env_config": _env_config_to_payload(env_config),
        "network_architecture": {
            "observation_dim": actor.observation_dim,
            "action_dim": actor.action_dim,
            "hidden_dim": actor.hidden_dim,
            "num_layers": actor.num_layers,
            "district_types": actor.district_types,
            "policy_arch": actor.policy_arch,
            "dueling": actor.dueling,
        },
        "q_network_state_dict": {
            key: value.detach().cpu()
            for key, value in actor.state_dict().items()
        },
        "obs_normalizer": obs_normalizer.state_dict() if obs_normalizer else None,
    }


def _init_parallel_eval_worker_from_context(context: dict[str, Any]) -> None:
    global _EVAL_CONTEXT
    env_config = _env_config_from_payload(context["env_config"])
    if "baseline_name" in context:
        baseline_name = context["baseline_name"]
        if baseline_name == "random":
            actor = RandomPhasePolicy(seed=context["seed"])
        elif baseline_name == "fixed":
            actor = FixedCyclePolicy(green_time=context["fixed_green_time"])
        elif baseline_name == "hold":
            actor = HoldPhasePolicy()
        elif baseline_name == "queue_greedy":
            actor = QueueGreedyPolicy()
        else:
            raise ValueError(f"Unsupported baseline worker kind: {baseline_name}")
        obs_normalizer = None
    else:
        architecture = context["network_architecture"]
        actor = TrafficControlQNetwork(
            observation_dim=architecture["observation_dim"],
            action_dim=architecture["action_dim"],
            hidden_dim=architecture["hidden_dim"],
            num_layers=architecture["num_layers"],
            district_types=tuple(architecture["district_types"]),
            policy_arch=architecture["policy_arch"],
            dueling=bool(architecture.get("dueling", True)),
        ).to(torch.device("cpu"))
        actor.load_state_dict(context["q_network_state_dict"])
        actor.eval()

        obs_normalizer = None
        if context.get("obs_normalizer"):
            obs_normalizer = RunningNormalizer()
            obs_normalizer.load_state_dict(context["obs_normalizer"])

    _EVAL_CONTEXT = {
        "env_config": env_config,
        "actor": actor,
        "obs_normalizer": obs_normalizer,
    }


def _parallel_eval_worker(
    scenario_spec: ScenarioSpec,
    index: int,
    worker_kind: str,
) -> dict[str, float | str]:
    del index, worker_kind
    env_config = _EVAL_CONTEXT["env_config"]
    actor = _EVAL_CONTEXT["actor"]
    obs_normalizer = _EVAL_CONTEXT["obs_normalizer"]

    return evaluate_policy(
        env_factory=lambda: TrafficEnv(
            city_id=scenario_spec.city_id,
            scenario_name=scenario_spec.scenario_name,
            city_dir=scenario_spec.city_dir,
            scenario_dir=scenario_spec.scenario_dir,
            config_path=scenario_spec.config_path,
            roadnet_path=scenario_spec.roadnet_path,
            district_map_path=scenario_spec.district_map_path,
            metadata_path=scenario_spec.metadata_path,
            env_config=env_config,
        ),
        actor=actor,
        device=torch.device("cpu"),
        obs_normalizer=obs_normalizer,
        deterministic=True,
    )


def _parallel_rollout_collection_worker(
    scenario_spec: ScenarioSpec,
    context: dict[str, Any],
    epsilon: float,
    max_decision_steps: int | None,
    gamma: float,
    n_step: int,
) -> dict[str, Any]:
    env_config = _env_config_from_payload(context["env_config"])
    architecture = context["network_architecture"]
    q_network = TrafficControlQNetwork(
        observation_dim=architecture["observation_dim"],
        action_dim=architecture["action_dim"],
        hidden_dim=architecture["hidden_dim"],
        num_layers=architecture["num_layers"],
        district_types=tuple(architecture["district_types"]),
        policy_arch=architecture["policy_arch"],
        dueling=bool(architecture.get("dueling", True)),
    ).to(torch.device("cpu"))
    q_network.load_state_dict(context["q_network_state_dict"])
    q_network.eval()

    obs_normalizer = None
    if context.get("obs_normalizer"):
        obs_normalizer = RunningNormalizer()
        obs_normalizer.load_state_dict(context["obs_normalizer"])

    env = TrafficEnv(
        city_id=scenario_spec.city_id,
        scenario_name=scenario_spec.scenario_name,
        city_dir=scenario_spec.city_dir,
        scenario_dir=scenario_spec.scenario_dir,
        config_path=scenario_spec.config_path,
        roadnet_path=scenario_spec.roadnet_path,
        district_map_path=scenario_spec.district_map_path,
        metadata_path=scenario_spec.metadata_path,
        env_config=env_config,
    )
    return _collect_episode_trajectory(
        env=env,
        q_network=q_network,
        obs_normalizer=obs_normalizer,
        epsilon=epsilon,
        max_decision_steps=max_decision_steps,
        gamma=gamma,
        n_step=n_step,
        device=torch.device("cpu"),
    )


def _collect_episode_trajectory(
    env: TrafficEnv,
    q_network: TrafficControlQNetwork,
    obs_normalizer: RunningNormalizer | None,
    epsilon: float,
    max_decision_steps: int | None,
    gamma: float,
    n_step: int,
    device: torch.device,
) -> dict[str, Any]:
    observation_batch = env.reset()
    n_step_buffers = [
        deque() for _ in range(len(observation_batch["intersection_ids"]))
    ]
    q_value_samples: list[float] = []
    transition_records: list[tuple[np.ndarray, int, np.ndarray, int, float, np.ndarray, int, np.ndarray, bool, float]] = []

    done = False
    decision_steps = 0
    last_info = env.last_info
    while not done:
        if max_decision_steps is not None and decision_steps >= max_decision_steps:
            break

        raw_obs = observation_batch["observations"].astype(np.float32)
        normalized_obs = obs_normalizer.normalize(raw_obs) if obs_normalizer else raw_obs
        obs_tensor = torch.as_tensor(normalized_obs, dtype=torch.float32, device=device)
        district_type_tensor = torch.as_tensor(
            observation_batch["district_type_indices"],
            dtype=torch.int64,
            device=device,
        )
        action_mask_tensor = torch.as_tensor(
            observation_batch["action_mask"],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            q_values = q_network.forward(
                observations=obs_tensor,
                district_type_indices=district_type_tensor,
                action_mask=action_mask_tensor,
            )
            actions = q_network.act(
                observations=obs_tensor,
                district_type_indices=district_type_tensor,
                action_mask=action_mask_tensor,
                deterministic=False,
                epsilon=epsilon,
            ).cpu().numpy()
        q_value_samples.append(float(q_values.max(dim=-1).values.mean().detach().cpu()))

        next_observation_batch, rewards, done, info = env.step(actions)
        transition_records.extend(
            _build_n_step_transitions(
                buffers=n_step_buffers,
                observation_batch=observation_batch,
                actions=actions,
                rewards=np.asarray(rewards, dtype=np.float32),
                next_observation_batch=next_observation_batch,
                done=done,
                gamma=gamma,
                n_step=n_step,
            )
        )
        observation_batch = next_observation_batch
        last_info = info
        decision_steps += 1

    transition_records.extend(
        _flush_n_step_transition_buffers(
            buffers=n_step_buffers,
            gamma=gamma,
        )
    )

    episode_metrics = {
        key: float(value)
        for key, value in last_info["metrics"].items()
        if value is not None and isinstance(value, (int, float))
    }
    episode_record = {
        **episode_metrics,
        "city_id": env.city_id,
        "scenario_name": env.scenario_name,
        "decision_steps": decision_steps,
        "transitions": len(transition_records),
        "episode_return": float(env.episode_return),
        "total_episode_return": float(env.total_episode_return),
        "epsilon": float(epsilon),
        "mean_q_value": float(np.mean(q_value_samples)) if q_value_samples else 0.0,
    }
    return {
        "episode_record": episode_record,
        "transitions": _pack_transition_records(transition_records, env.observation_dim),
    }


def _build_n_step_transitions(
    buffers: list[deque[StepRecord]],
    observation_batch: dict[str, Any],
    actions: np.ndarray,
    rewards: np.ndarray,
    next_observation_batch: dict[str, Any],
    done: bool,
    gamma: float,
    n_step: int,
) -> list[tuple[np.ndarray, int, np.ndarray, int, float, np.ndarray, int, np.ndarray, bool, float]]:
    transition_records: list[tuple[np.ndarray, int, np.ndarray, int, float, np.ndarray, int, np.ndarray, bool, float]] = []
    for row_index, buffer in enumerate(buffers):
        record = StepRecord(
            observation=observation_batch["observations"][row_index].astype(np.float32),
            district_type_index=int(observation_batch["district_type_indices"][row_index]),
            action_mask=observation_batch["action_mask"][row_index].astype(np.float32),
            action=int(actions[row_index]),
            reward=float(rewards[row_index]),
            next_observation=next_observation_batch["observations"][row_index].astype(np.float32),
            next_district_type_index=int(next_observation_batch["district_type_indices"][row_index]),
            next_action_mask=next_observation_batch["action_mask"][row_index].astype(np.float32),
            done=bool(done),
        )
        buffer.append(record)
        if len(buffer) >= n_step:
            transition_records.append(_make_transition_from_buffer(buffer, steps=n_step, gamma=gamma))
            buffer.popleft()
    return transition_records


def _flush_n_step_transition_buffers(
    buffers: list[deque[StepRecord]],
    gamma: float,
) -> list[tuple[np.ndarray, int, np.ndarray, int, float, np.ndarray, int, np.ndarray, bool, float]]:
    transition_records: list[tuple[np.ndarray, int, np.ndarray, int, float, np.ndarray, int, np.ndarray, bool, float]] = []
    for buffer in buffers:
        while buffer:
            transition_records.append(
                _make_transition_from_buffer(buffer, steps=len(buffer), gamma=gamma)
            )
            buffer.popleft()
    return transition_records


def _make_transition_from_buffer(
    buffer: deque[StepRecord],
    steps: int,
    gamma: float,
) -> tuple[np.ndarray, int, np.ndarray, int, float, np.ndarray, int, np.ndarray, bool, float]:
    records = list(islice(buffer, 0, steps))
    reward = 0.0
    for step_index, record in enumerate(records):
        reward += (gamma ** step_index) * float(record.reward)
    first_record = records[0]
    last_record = records[-1]
    discount = gamma ** len(records)
    return (
        first_record.observation,
        first_record.district_type_index,
        first_record.action_mask,
        first_record.action,
        reward,
        last_record.next_observation,
        last_record.next_district_type_index,
        last_record.next_action_mask,
        last_record.done,
        discount,
    )


def _pack_transition_records(
    transition_records: list[tuple[np.ndarray, int, np.ndarray, int, float, np.ndarray, int, np.ndarray, bool, float]],
    observation_dim: int,
) -> dict[str, np.ndarray]:
    if not transition_records:
        return {
            "observations": np.zeros((0, observation_dim), dtype=np.float32),
            "district_type_indices": np.zeros(0, dtype=np.int64),
            "action_masks": np.zeros((0, 2), dtype=np.float32),
            "actions": np.zeros(0, dtype=np.int64),
            "rewards": np.zeros(0, dtype=np.float32),
            "next_observations": np.zeros((0, observation_dim), dtype=np.float32),
            "next_district_type_indices": np.zeros(0, dtype=np.int64),
            "next_action_masks": np.zeros((0, 2), dtype=np.float32),
            "dones": np.zeros(0, dtype=np.float32),
            "discounts": np.zeros(0, dtype=np.float32),
        }

    observations = np.stack([record[0] for record in transition_records]).astype(np.float32)
    district_type_indices = np.asarray([record[1] for record in transition_records], dtype=np.int64)
    action_masks = np.stack([record[2] for record in transition_records]).astype(np.float32)
    actions = np.asarray([record[3] for record in transition_records], dtype=np.int64)
    rewards = np.asarray([record[4] for record in transition_records], dtype=np.float32)
    next_observations = np.stack([record[5] for record in transition_records]).astype(np.float32)
    next_district_type_indices = np.asarray([record[6] for record in transition_records], dtype=np.int64)
    next_action_masks = np.stack([record[7] for record in transition_records]).astype(np.float32)
    dones = np.asarray([record[8] for record in transition_records], dtype=np.float32)
    discounts = np.asarray([record[9] for record in transition_records], dtype=np.float32)
    return {
        "observations": observations,
        "district_type_indices": district_type_indices,
        "action_masks": action_masks,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "next_district_type_indices": next_district_type_indices,
        "next_action_masks": next_action_masks,
        "dones": dones,
        "discounts": discounts,
    }
