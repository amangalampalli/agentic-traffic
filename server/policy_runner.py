"""Run traffic policies and generate CityFlow replay files for visualization."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import gc
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
_district_llm_inference = None

LLM_MODEL_PATH = Path(
    os.environ.get("LLM_MODEL_PATH", "")
    or (REPO_ROOT / "artifacts" / "district_llm_adapter_v3" / "main_run" / "adapter")
)
VISUALIZER_MAX_SIM_SECONDS = int(os.environ.get("VISUALIZER_MAX_SIM_SECONDS", "180"))


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

ALL_POLICIES = (
    "no_intervention",
    "fixed",
    "random",
    "learned",
    "dqn_heuristic",
    "llm_dqn",
)


class _LoadedDQNPolicyAdapter:
    @property
    def env_config(self):
        return _dqn_env_config

    def decide(self, observation_batch: dict[str, Any]):
        from district_llm.rl_guidance_wrapper import RLPolicyDecision

        if _dqn_actor is None:
            raise RuntimeError("DQN checkpoint not loaded. Call load_dqn_checkpoint() first.")

        raw_obs = observation_batch["observations"].astype(np.float32)
        normalized_obs = (
            _dqn_obs_normalizer.normalize(raw_obs) if _dqn_obs_normalizer is not None else raw_obs
        )
        obs_tensor = torch.as_tensor(normalized_obs, dtype=torch.float32)
        district_type_tensor = torch.as_tensor(
            observation_batch["district_type_indices"],
            dtype=torch.int64,
        )
        action_mask_tensor = torch.as_tensor(
            observation_batch["action_mask"],
            dtype=torch.float32,
        )
        with torch.no_grad():
            q_values = _dqn_actor.forward(
                observations=obs_tensor,
                district_type_indices=district_type_tensor,
                action_mask=action_mask_tensor,
            )
        q_values_np = q_values.detach().cpu().numpy().astype(np.float32)
        return RLPolicyDecision(
            q_values=q_values_np,
            actions=q_values_np.argmax(axis=1).astype(np.int64),
        )


def _load_district_llm_inference():
    global _district_llm_inference
    if _district_llm_inference is not None:
        return _district_llm_inference
    if not LLM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LLM adapter path not found: {LLM_MODEL_PATH}. "
            "Set LLM_MODEL_PATH to enable the llm_dqn visualizer policy."
        )
    from district_llm.inference import DistrictLLMInference
    from district_llm.repair import RepairConfig

    _district_llm_inference = DistrictLLMInference(
        model_name_or_path=str(LLM_MODEL_PATH),
        device=None,
        repair_config=RepairConfig(
            allow_only_visible_candidates=True,
            max_target_intersections=3,
            fallback_on_empty_targets=True,
            fallback_mode="heuristic",
        ),
    )
    return _district_llm_inference


def load_district_llm_inference():
    inference = _load_district_llm_inference()
    print(f"[policy_runner] District LLM prewarmed from {LLM_MODEL_PATH}")
    return inference


def unload_district_llm_inference() -> None:
    global _district_llm_inference
    if _district_llm_inference is None:
        return
    inference = _district_llm_inference
    _district_llm_inference = None

    model = getattr(inference, "model", None)
    tokenizer = getattr(inference, "tokenizer", None)
    if model is not None:
        try:
            del model
        except Exception:
            pass
    if tokenizer is not None:
        try:
            del tokenizer
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    print("[policy_runner] District LLM unloaded")


def _build_guided_controller(policy_name: str):
    from district_llm.heuristic_guidance import HeuristicGuidanceConfig
    from district_llm.rl_guidance_wrapper import (
        DistrictGuidedRLController,
        GuidanceInfluenceConfig,
        HeuristicGuidanceProvider,
        LLMGuidanceProvider,
    )
    from district_llm.summary_builder import DistrictStateSummaryBuilder

    heuristic_provider = HeuristicGuidanceProvider(
        config=HeuristicGuidanceConfig(max_target_intersections=3)
    )
    influence_config = GuidanceInfluenceConfig(
        wrapper_mode="target_only_soft",
        bias_strength=0.025,
        target_only_bias_strength=0.025,
        corridor_bias_strength=0.0125,
        max_intersections_affected=2,
        guidance_refresh_steps=10,
        guidance_persistence_steps=5,
        max_guidance_duration=10,
        apply_global_bias=False,
        apply_target_only=True,
        gating_mode="queue_or_imbalance",
        min_avg_queue_for_guidance=150.0,
        min_queue_imbalance_for_guidance=20.0,
        require_incident_or_spillback=False,
        allow_guidance_in_normal_conditions=False,
        enable_bias_decay=False,
        bias_decay_schedule="linear",
        fallback_policy="no_op",
        log_guidance_debug=False,
    ).validate()
    summary_builder = DistrictStateSummaryBuilder(top_k=3, candidate_limit=6)
    guidance_provider = heuristic_provider
    mode_source = "rl_heuristic"
    if policy_name == "llm_dqn":
        guidance_provider = LLMGuidanceProvider(
            inference=_load_district_llm_inference(),
            max_new_tokens=128,
        )
        mode_source = "rl_llm"
    print(
        f"[policy_runner] guided_controller policy={policy_name} mode_source={mode_source} "
        f"wrapper_mode={influence_config.wrapper_mode} bias={influence_config.bias_strength} "
        f"target_bias={influence_config.target_only_bias_strength} corridor_bias={influence_config.corridor_bias_strength} "
        f"max_affected={influence_config.max_intersections_affected} gating={influence_config.gating_mode} "
        f"refresh={influence_config.guidance_refresh_steps} persistence={influence_config.guidance_persistence_steps} "
        f"fallback_policy={influence_config.fallback_policy}"
    )
    return DistrictGuidedRLController(
        policy=_LoadedDQNPolicyAdapter(),
        mode_source=mode_source,
        summary_builder=summary_builder,
        guidance_provider=guidance_provider,
        influence_config=influence_config,
        heuristic_provider=heuristic_provider,
    )


def _evaluate_guided_policy(env_factory, controller) -> dict[str, float | str]:
    env = env_factory()
    observation_batch = env.reset()
    done = False
    final_info = env.last_info
    controller.reset()
    max_decision_steps = max(
        1,
        int(getattr(env, "max_episode_seconds", 0) // max(1, env.env_config.decision_interval)),
    )

    while not done:
        action_batch = controller.act(env=env, observation_batch=observation_batch)
        observation_batch, _, done, final_info = env.step(action_batch.actions)
        decision_step = int(getattr(env, "decision_step_count", 0))
        should_log = decision_step == 1 or done or (decision_step % 5 == 0)
        if should_log:
            metrics = final_info.get("metrics", {}) if isinstance(final_info, dict) else {}
            print(
                f"[policy_runner][{controller.mode_source}] step={decision_step}/{max_decision_steps} "
                f"sim_time={int(env.adapter.get_current_time())}s "
                f"wait={float(metrics.get('mean_waiting_vehicles', float('nan'))):.2f} "
                f"throughput={float(metrics.get('throughput', float('nan'))):.1f}"
            )

    metrics = {
        key: float(value)
        for key, value in final_info["metrics"].items()
        if value is not None and isinstance(value, (int, float))
    }
    metrics.update(
        {
            "city_id": env.city_id,
            "scenario_name": env.scenario_name,
            "episode_return": float(env.episode_return),
            "total_episode_return": float(env.total_episode_return),
            "decision_steps": float(env.decision_step_count),
        }
    )
    metrics.update(controller.episode_debug_summary())
    return metrics


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
    from training.cityflow_dataset import CityFlowDataset, ScenarioSpec
    from training.rollout import evaluate_policy
    from training.train_local_policy import build_env

    output_dir = output_root / city_id / scenario_name / policy_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[policy_runner] start policy={policy_name} city={city_id} scenario={scenario_name} "
        f"max_sim_seconds={VISUALIZER_MAX_SIM_SECONDS}"
    )

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
    original_step = int(temp_config.get("step", 0) or 0)
    temp_config["step"] = (
        VISUALIZER_MAX_SIM_SECONDS if original_step <= 0
        else min(original_step, VISUALIZER_MAX_SIM_SECONDS)
    )

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
        elif policy_name in {"dqn_heuristic", "llm_dqn"}:
            actor = _build_guided_controller(policy_name)
            device = None
            obs_normalizer = None
        else:
            raise ValueError(f"Unknown policy name: {policy_name!r}")

        if policy_name in {"dqn_heuristic", "llm_dqn"}:
            metrics = _evaluate_guided_policy(
                env_factory=lambda: build_env(env_config, temp_spec),
                controller=actor,
            )
        else:
            metrics = evaluate_policy(
                env_factory=lambda: build_env(env_config, temp_spec),
                actor=actor,
                device=device,
                obs_normalizer=obs_normalizer,
                deterministic=True,
                log_prefix=f"[policy_runner][{policy_name}]",
                log_every_steps=5,
            )
    finally:
        temp_config_path.unlink(missing_ok=True)

    # Persist metrics so subsequent requests can be served from cache.
    (output_dir / "metrics.json").write_text(json.dumps(metrics))
    print(
        f"[policy_runner] done policy={policy_name} city={city_id} scenario={scenario_name} "
        f"decision_steps={metrics.get('decision_steps')} replay={replay_path.exists()} "
        f"roadnet_log={roadnet_log_path.exists()}"
    )

    return RunResult(
        policy_name=policy_name,
        metrics=metrics,
        replay_path=replay_path,
        roadnet_log_path=roadnet_log_path,
    )
