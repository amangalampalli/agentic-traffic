from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.local_policy import FixedCyclePolicy, RandomPhasePolicy
from training.cityflow_dataset import CityFlowDataset
from training.device import configure_torch_runtime, resolve_torch_device
from training.models import RunningNormalizer, TrafficControlQNetwork
from training.rollout import evaluate_policy
from training.train_local_policy import build_env, build_env_config, load_env_config
from training.trainer import aggregate_metrics, aggregate_metrics_by_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a learned local policy checkpoint against fixed and random "
            "baselines under the same reward config."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--city-id", default=None)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-val-cities", type=int, default=None)
    parser.add_argument("--scenarios-per-city", type=int, default=1)
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--device", default=None)

    parser.add_argument("--decision-interval", type=int, default=5)
    parser.add_argument("--simulator-interval", type=int, default=1)
    parser.add_argument("--min-green-time", type=int, default=10)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--max-episode-seconds", type=int, default=None)
    parser.add_argument("--max-incoming-lanes", type=int, default=16)
    parser.add_argument("--count-scale", type=float, default=20.0)
    parser.add_argument("--elapsed-time-scale", type=float, default=60.0)
    parser.add_argument("--disable-district-context", action="store_true")
    parser.add_argument("--disable-outgoing-congestion", action="store_true")
    parser.add_argument("--reward-variant", default="wait_queue_throughput")
    parser.add_argument("--waiting-weight", type=float, default=1.0)
    parser.add_argument("--vehicle-weight", type=float, default=0.1)
    parser.add_argument("--pressure-weight", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--disable-lane-reward-normalization", action="store_true")
    parser.add_argument("--reward-clip", type=float, default=5.0)
    parser.add_argument("--queue-delta-weight", type=float, default=2.0)
    parser.add_argument("--wait-delta-weight", type=float, default=4.0)
    parser.add_argument("--queue-level-weight", type=float, default=0.5)
    parser.add_argument("--wait-level-weight", type=float, default=1.0)
    parser.add_argument("--throughput-weight", type=float, default=0.1)
    parser.add_argument("--imbalance-weight", type=float, default=0.1)
    parser.add_argument("--reward-delta-clip", type=float, default=2.0)
    parser.add_argument("--reward-level-normalizer", type=float, default=10.0)
    parser.add_argument("--throughput-normalizer", type=float, default=2.0)
    parser.add_argument("--policy-arch", default="single_head_with_district_feature")
    parser.add_argument("--fixed-green-time", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--verbose-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.city_id is None) != (args.scenario_name is None):
        raise ValueError("--city-id and --scenario-name must be provided together.")

    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    scenario_specs = build_scenario_specs(dataset, args)

    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    print(f"[setup] torch_device={device.type}")

    env_config = build_env_config(args)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("env_config"):
        env_config = load_env_config(checkpoint["env_config"])

    network_architecture = checkpoint.get("network_architecture") or checkpoint.get(
        "policy_architecture", {}
    )
    trainer_config = checkpoint.get("dqn_config", {})
    checkpoint_policy_arch = network_architecture.get(
        "policy_arch",
        trainer_config.get("policy_arch", args.policy_arch),
    )

    actor = TrafficControlQNetwork(
        observation_dim=int(network_architecture["observation_dim"]),
        action_dim=int(network_architecture.get("action_dim", 2)),
        hidden_dim=int(trainer_config.get("hidden_dim", 256)),
        num_layers=int(trainer_config.get("hidden_layers", 2)),
        district_types=tuple(network_architecture.get("district_types", ())),
        policy_arch=checkpoint_policy_arch,
        dueling=bool(network_architecture.get("dueling", True)),
    ).to(device)
    actor.load_state_dict(
        checkpoint.get("q_network_state_dict") or checkpoint["policy_state_dict"]
    )
    actor.eval()

    obs_normalizer = None
    if checkpoint.get("obs_normalizer"):
        obs_normalizer = RunningNormalizer()
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

    policies = {
        "learned": (actor, device, obs_normalizer),
        "fixed": (FixedCyclePolicy(green_time=args.fixed_green_time), None, None),
        "random": (RandomPhasePolicy(seed=args.random_seed), None, None),
    }
    scope = build_scope_summary(args, scenario_specs)
    print(
        "[compare] "
        f"num_cities={scope['num_cities']} "
        f"num_scenarios={scope['num_scenarios']} "
        f"reward_variant={env_config.reward.variant}"
    )

    aggregate_results: dict[str, dict[str, float]] = {}
    scenario_breakdowns: dict[str, dict[str, float]] = {}
    for name, (policy, policy_device, normalizer) in policies.items():
        print(f"[compare] starting policy={name}")
        episode_metrics = []
        iterator = enumerate(scenario_specs, start=1)
        if not args.disable_tqdm:
            iterator = tqdm(
                iterator,
                total=len(scenario_specs),
                desc=f"compare:{name}",
                dynamic_ncols=True,
                leave=False,
            )
        for index, spec in iterator:
            if args.verbose_progress:
                message = (
                    f"[compare] policy={name} city={spec.city_id} "
                    f"scenario={spec.scenario_name} i={index}/{len(scenario_specs)}"
                )
                if args.disable_tqdm:
                    print(message)
                else:
                    tqdm.write(message)
            metrics = evaluate_policy(
                env_factory=lambda spec=spec, config=env_config: build_env(config, spec),
                actor=policy,
                device=policy_device,
                obs_normalizer=normalizer,
                deterministic=True,
            )
            episode_metrics.append(metrics)
            if not args.disable_tqdm:
                iterator.set_postfix(
                    city=spec.city_id,
                    scenario=spec.scenario_name,
                    ret=f"{metrics['episode_return']:.3f}",
                )
        aggregate_results[name] = aggregate_metrics(episode_metrics)
        scenario_breakdowns[name] = aggregate_metrics_by_scenario(episode_metrics)
        mean_return = aggregate_results[name].get("mean_episode_return", float("nan"))
        mean_wait = aggregate_results[name].get("mean_mean_waiting_vehicles", float("nan"))
        mean_throughput = aggregate_results[name].get("mean_throughput", float("nan"))
        message = (
            f"[compare] finished policy={name} "
            f"mean_return={mean_return:.3f} "
            f"wait={mean_wait:.3f} "
            f"throughput={mean_throughput:.1f}"
        )
        if args.disable_tqdm:
            print(message)
        else:
            tqdm.write(message)

    learned = aggregate_results["learned"]
    fixed = aggregate_results["fixed"]
    random = aggregate_results["random"]
    summary = {
        "comparison_scope": build_scope_summary(args, scenario_specs),
        "reward_variant": env_config.reward.variant,
        "checkpoint": args.checkpoint,
        "results": aggregate_results,
        "scenario_breakdowns": scenario_breakdowns,
        "deltas": {
            "learned_minus_fixed_return": float(learned["mean_episode_return"])
            - float(fixed["mean_episode_return"]),
            "learned_minus_random_return": float(learned["mean_episode_return"])
            - float(random["mean_episode_return"]),
            "learned_minus_fixed_wait": float(learned["mean_mean_waiting_vehicles"])
            - float(fixed["mean_mean_waiting_vehicles"]),
            "learned_minus_random_wait": float(learned["mean_mean_waiting_vehicles"])
            - float(random["mean_mean_waiting_vehicles"]),
            "learned_minus_fixed_travel_time": float(learned["mean_average_travel_time"])
            - float(fixed["mean_average_travel_time"]),
            "learned_minus_random_travel_time": float(learned["mean_average_travel_time"])
            - float(random["mean_average_travel_time"]),
            "learned_minus_fixed_throughput": float(learned["mean_throughput"])
            - float(fixed["mean_throughput"]),
            "learned_minus_random_throughput": float(learned["mean_throughput"])
            - float(random["mean_throughput"]),
        },
    }
    print(json.dumps(summary, indent=2))


def build_scenario_specs(dataset: CityFlowDataset, args: argparse.Namespace) -> list:
    if args.city_id and args.scenario_name:
        return [dataset.build_scenario_spec(args.city_id, args.scenario_name)]
    return dataset.iter_scenarios(
        split_name=args.split,
        scenarios_per_city=args.scenarios_per_city,
        max_cities=args.max_val_cities,
        diversify_single_scenario=True,
    )


def build_scope_summary(args: argparse.Namespace, scenario_specs: list) -> dict[str, object]:
    city_ids = sorted({spec.city_id for spec in scenario_specs})
    scenario_names = sorted({spec.scenario_name for spec in scenario_specs})
    return {
        "split": args.split if not args.city_id else None,
        "city_id": args.city_id,
        "scenario_name": args.scenario_name,
        "num_cities": len(city_ids),
        "num_scenarios": len(scenario_specs),
        "city_ids": city_ids,
        "scenario_names": scenario_names,
    }


if __name__ == "__main__":
    main()
