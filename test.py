from __future__ import annotations

import argparse
import json
import random

import numpy as np

from env.observation_builder import ObservationConfig
from env.reward import RewardConfig
from env.traffic_env import EnvConfig, TrafficEnv
from training.dataset import CityFlowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for the CityFlow RL environment.")
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--city-id", default=None)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--decision-steps", type=int, default=5)
    parser.add_argument("--decision-interval", type=int, default=5)
    parser.add_argument("--min-green-time", type=int, default=10)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    dataset.generate_default_splits()
    scenario_spec = (
        dataset.build_scenario_spec(args.city_id, args.scenario_name)
        if args.city_id and args.scenario_name
        else dataset.sample_scenario("train", rng)
    )

    env = TrafficEnv(
        city_id=scenario_spec.city_id,
        scenario_name=scenario_spec.scenario_name,
        city_dir=scenario_spec.city_dir,
        scenario_dir=scenario_spec.scenario_dir,
        config_path=scenario_spec.config_path,
        roadnet_path=scenario_spec.roadnet_path,
        district_map_path=scenario_spec.district_map_path,
        metadata_path=scenario_spec.metadata_path,
        env_config=EnvConfig(
            decision_interval=args.decision_interval,
            min_green_time=args.min_green_time,
            thread_num=args.thread_num,
            observation=ObservationConfig(),
            reward=RewardConfig(),
        ),
    )

    observation_batch = env.reset()
    print(
        json.dumps(
            {
                "city_id": env.city_id,
                "scenario_name": env.scenario_name,
                "num_controlled_intersections": len(observation_batch["intersection_ids"]),
                "observation_shape": list(observation_batch["observations"].shape),
                "lane_mask_shape": list(observation_batch["lane_mask"].shape),
                "observation_dim": env.observation_dim,
            },
            indent=2,
        )
    )

    for decision_step in range(args.decision_steps):
        random_actions = np.asarray(
            [rng.randint(0, 1) for _ in observation_batch["intersection_ids"]],
            dtype=np.int64,
        )
        observation_batch, rewards, done, info = env.step(random_actions)
        reward_summary = {
            "decision_step": decision_step + 1,
            "reward_mean": float(rewards.mean()),
            "reward_min": float(rewards.min()),
            "reward_max": float(rewards.max()),
            "mean_waiting_vehicles": info["metrics"]["mean_waiting_vehicles"],
            "throughput": info["metrics"]["throughput"],
            "sim_time": info["sim_time"],
        }
        print(json.dumps(reward_summary))
        if done:
            break


if __name__ == "__main__":
    main()
