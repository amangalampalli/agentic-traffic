"""Run traffic policies and generate CityFlow replay files for visualization."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- DQN singleton (loaded once at server startup) ---

_dqn_actor = None
_dqn_obs_normalizer = None
_dqn_env_config = None


def load_dqn_checkpoint(checkpoint_path: str | Path) -> None:
    global _dqn_actor, _dqn_obs_normalizer, _dqn_env_config

    from training.models import RunningNormalizer, TrafficControlQNetwork
    from training.train_local_policy import load_env_config

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    network_arch = checkpoint.get("network_architecture") or checkpoint.get(
        "policy_architecture", {}
    )
    trainer_config = checkpoint.get("dqn_config", {})
    policy_arch = network_arch.get(
        "policy_arch", trainer_config.get("policy_arch", "single_head")
    )

    actor = TrafficControlQNetwork(
        observation_dim=int(network_arch["observation_dim"]),
        action_dim=int(network_arch.get("action_dim", 2)),
        hidden_dim=int(trainer_config.get("hidden_dim", 256)),
        num_layers=int(trainer_config.get("hidden_layers", 2)),
        district_types=tuple(network_arch.get("district_types", ())),
        policy_arch=policy_arch,
        dueling=bool(network_arch.get("dueling", True)),
    )
    actor.load_state_dict(
        checkpoint.get("q_network_state_dict") or checkpoint["policy_state_dict"]
    )
    actor.eval()

    obs_normalizer = None
    if checkpoint.get("obs_normalizer"):
        obs_normalizer = RunningNormalizer()
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

    env_config = None
    if checkpoint.get("env_config"):
        env_config = load_env_config(checkpoint["env_config"])

    _dqn_actor = actor
    _dqn_obs_normalizer = obs_normalizer
    _dqn_env_config = env_config
    print(f"[policy_runner] DQN checkpoint loaded from {Path(checkpoint_path).name}")


# --- Result type ---

@dataclass
class RunResult:
    policy_name: str
    metrics: dict[str, Any]
    replay_path: Path
    roadnet_log_path: Path


# --- Core runner ---

ALL_POLICIES = ("no_intervention", "fixed", "random", "learned")


def run_policy_for_city(
    city_id: str,
    scenario_name: str,
    policy_name: str,
    generated_root: Path,
    output_root: Path,
) -> RunResult:
    """Run a single policy on one city/scenario and write a CityFlow replay file."""
    from agents.local_policy import FixedCyclePolicy, HoldPhasePolicy, RandomPhasePolicy
    from env.traffic_env import EnvConfig
    from training.dataset import CityFlowDataset, ScenarioSpec
    from training.rollout import evaluate_policy
    from training.train_local_policy import build_env

    output_dir = output_root / city_id / scenario_name / policy_name
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_path = output_dir / "replay.txt"
    roadnet_log_path = output_dir / "roadnetLogFile.json"

    dataset = CityFlowDataset(generated_root=str(generated_root))
    spec = dataset.build_scenario_spec(city_id, scenario_name)

    # Build a modified config that enables replay to our output dir.
    original_config = json.loads(spec.config_path.read_text())
    city_dir_resolved = spec.city_dir.resolve()

    # Compute replay/roadnet paths relative to the city dir (CityFlow resolves from dir).
    rel_replay = os.path.relpath(
        str(replay_path.resolve()), str(city_dir_resolved)
    ).replace("\\", "/")
    rel_roadnet_log = os.path.relpath(
        str(roadnet_log_path.resolve()), str(city_dir_resolved)
    ).replace("\\", "/")

    temp_config = dict(original_config)
    temp_config["saveReplay"] = True
    temp_config["replayLogFile"] = rel_replay
    temp_config["roadnetLogFile"] = rel_roadnet_log

    # Write temp config to a temporary file so we don't touch on-disk configs.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=str(output_dir)
    ) as tmp:
        json.dump(temp_config, tmp)
        temp_config_path = Path(tmp.name)

    try:
        temp_spec = ScenarioSpec(
            city_id=spec.city_id,
            scenario_name=spec.scenario_name,
            city_dir=spec.city_dir,
            scenario_dir=spec.scenario_dir,
            config_path=temp_config_path,
            roadnet_path=spec.roadnet_path,
            district_map_path=spec.district_map_path,
            metadata_path=spec.metadata_path,
        )

        env_config = _dqn_env_config or EnvConfig()

        if policy_name == "learned":
            if _dqn_actor is None:
                raise RuntimeError("DQN checkpoint not loaded. Call load_dqn_checkpoint() first.")
            actor = _dqn_actor
            device = None
            obs_normalizer = _dqn_obs_normalizer
        elif policy_name == "fixed":
            actor = FixedCyclePolicy(green_time=20)
            device = None
            obs_normalizer = None
        elif policy_name == "random":
            actor = RandomPhasePolicy(seed=7)
            device = None
            obs_normalizer = None
        elif policy_name == "no_intervention":
            actor = HoldPhasePolicy()
            device = None
            obs_normalizer = None
        else:
            raise ValueError(f"Unknown policy name: {policy_name!r}")

        metrics = evaluate_policy(
            env_factory=lambda: build_env(env_config, temp_spec),
            actor=actor,
            device=device,
            obs_normalizer=obs_normalizer,
            deterministic=True,
        )
    finally:
        temp_config_path.unlink(missing_ok=True)

    return RunResult(
        policy_name=policy_name,
        metrics=metrics,
        replay_path=replay_path,
        roadnet_log_path=roadnet_log_path,
    )
