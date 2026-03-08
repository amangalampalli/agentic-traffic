from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from agents.local_policy import FixedCyclePolicy, HoldPhasePolicy
from training.cityflow_dataset import CityFlowDataset
from training.train_local_policy import build_env, build_env_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test the CityFlow environment and district-type routing."
    )
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--city-id", default="city_0001")
    parser.add_argument("--scenario-name", default="normal")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--policy", choices=("hold", "fixed", "random"), default="random")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decision-interval", type=int, default=5)
    parser.add_argument("--simulator-interval", type=int, default=1)
    parser.add_argument("--min-green-time", type=int, default=10)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--max-episode-seconds", type=int, default=120)
    parser.add_argument("--max-incoming-lanes", type=int, default=16)
    parser.add_argument("--count-scale", type=float, default=20.0)
    parser.add_argument("--elapsed-time-scale", type=float, default=60.0)
    parser.add_argument("--disable-district-context", action="store_true")
    parser.add_argument("--disable-outgoing-congestion", action="store_true")
    parser.add_argument("--waiting-weight", type=float, default=1.0)
    parser.add_argument("--vehicle-weight", type=float, default=0.25)
    parser.add_argument("--pressure-weight", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    scenario_spec = dataset.build_scenario_spec(args.city_id, args.scenario_name)
    env = build_env(build_env_config(args), scenario_spec)

    observation = env.reset()
    print(
        json.dumps(
            {
                "city_id": args.city_id,
                "scenario_name": args.scenario_name,
                "observation_shape": list(observation["observations"].shape),
                "observation_dim": env.observation_dim,
                "controlled_intersections": len(observation["intersection_ids"]),
                "district_type_counts": Counter(observation["district_types"]),
                "district_type_indices_sample": observation["district_type_indices"][:10].tolist(),
                "boundary_fraction": float(observation["boundary_mask"].mean()),
            },
            indent=2,
        )
    )

    policy = resolve_policy(args.policy)
    for step in range(args.steps):
        if args.policy == "random":
            actions = sample_random_actions(observation["action_mask"], rng)
        else:
            actions = policy.act(observation)

        observation, rewards, done, info = env.step(actions)
        print(
            json.dumps(
                {
                    "step": step,
                    "sim_time": info["sim_time"],
                    "reward_mean": float(rewards.mean()),
                    "reward_min": float(rewards.min()),
                    "reward_max": float(rewards.max()),
                    "waiting_mean": info["metrics"]["mean_waiting_vehicles"],
                    "throughput": info["metrics"]["throughput"],
                    "district_type_metrics": {
                        key: value
                        for key, value in info["metrics"].items()
                        if "residential" in key
                        or "commercial" in key
                        or "industrial" in key
                        or "mixed" in key
                    },
                },
                indent=2,
            )
        )
        if done:
            break


def resolve_policy(name: str):
    if name == "hold":
        return HoldPhasePolicy()
    if name == "fixed":
        return FixedCyclePolicy()
    return None


def sample_random_actions(action_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    actions = np.zeros(action_mask.shape[0], dtype=np.int64)
    for row_index, mask in enumerate(action_mask):
        valid_actions = np.flatnonzero(mask > 0.0)
        actions[row_index] = int(rng.choice(valid_actions))
    return actions


if __name__ == "__main__":
    main()
