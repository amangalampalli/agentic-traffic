from __future__ import annotations

import argparse
import json
import os

from agents.local_policy import FixedCyclePolicy, HoldPhasePolicy, QueueGreedyPolicy, RandomPhasePolicy
from env.intersection_config import DISTRICT_TYPES
from env.observation_builder import ObservationConfig
from env.reward import REWARD_VARIANTS, RewardConfig
from env.traffic_env import EnvConfig
from training.dataset import CityFlowDataset
from training.models import POLICY_ARCHES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parameter-shared DQN for CityFlow traffic control."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_splits = subparsers.add_parser("make-splits", help="Generate city-level train/val/test splits.")
    make_splits.add_argument("--generated-root", default="data/generated")
    make_splits.add_argument("--splits-root", default="data/splits")
    make_splits.add_argument("--seed", type=int, default=7)
    make_splits.add_argument("--train-ratio", type=float, default=0.7)
    make_splits.add_argument("--val-ratio", type=float, default=0.15)
    make_splits.add_argument("--overwrite", action="store_true")

    train = subparsers.add_parser(
        "train",
        help="Train parameter-shared DQN with district-aware architecture variants.",
    )
    add_common_dataset_args(train)
    add_common_env_args(train)
    train.add_argument("--output-dir", default="artifacts/dqn_shared")
    train.add_argument("--policy-arch", choices=POLICY_ARCHES, default="single_head_with_district_feature")
    train.add_argument("--total-updates", type=int, default=200)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--n-step", type=int, default=3)
    train.add_argument("--replay-capacity", type=int, default=500000)
    train.add_argument("--minibatch-size", type=int, default=1024)
    train.add_argument("--learning-starts", type=int, default=10000)
    train.add_argument("--gradient-steps", type=int, default=64)
    train.add_argument("--target-tau", type=float, default=0.01)
    train.add_argument("--max-grad-norm", type=float, default=10.0)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--hidden-layers", type=int, default=2)
    train.add_argument("--disable-dueling", action="store_true")
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--eval-every", type=int, default=40)
    train.add_argument("--checkpoint-every", type=int, default=40)
    train.add_argument("--disable-checkpoint-on-eval", action="store_true")
    train.add_argument("--val-scenarios-per-city", type=int, default=1)
    train.add_argument("--max-val-cities", type=int, default=5)
    train.add_argument("--max-train-cities", type=int, default=None)
    train.add_argument("--num-rollout-workers", type=int, default=4)
    train.add_argument("--rollout-episodes-per-update", type=int, default=None)
    train.add_argument("--train-city-id", default=None)
    train.add_argument("--train-scenario-name", default=None)
    train.add_argument("--overfit-val-on-train-scenario", action="store_true")
    train.add_argument("--rollout-decision-steps", type=int, default=256)
    train.add_argument("--resume-from", default=None)
    train.add_argument("--disable-obs-norm", action="store_true")
    train.add_argument("--epsilon-start", type=float, default=1.0)
    train.add_argument("--epsilon-end", type=float, default=0.05)
    train.add_argument("--epsilon-decay-steps", type=int, default=50000)
    train.add_argument("--prioritized-replay-alpha", type=float, default=0.6)
    train.add_argument("--prioritized-replay-beta-start", type=float, default=0.4)
    train.add_argument("--prioritized-replay-beta-end", type=float, default=1.0)
    train.add_argument("--prioritized-replay-beta-steps", type=int, default=200000)
    train.add_argument("--rolling-window-size", type=int, default=20)
    train.add_argument("--disable-baseline-comparison", action="store_true")
    train.add_argument("--fail-on-val-error", action="store_true")
    train.add_argument("--verbose-progress", action="store_true")
    train.add_argument("--debug-fast", action="store_true")
    train.add_argument("--fast-overfit", action="store_true")
    train.add_argument(
        "--eval-num-workers",
        type=int,
        default=-1,
        help="CPU workers for validation. -1 uses all CPU cores; learned and baseline eval both parallelize across CPU workers.",
    )
    train.add_argument("--disable-tensorboard", action="store_true")
    train.add_argument("--disable-tqdm", action="store_true")
    train.add_argument("--tensorboard-log-dir", default=None)
    train.add_argument("--device", default=None)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint or baseline policy.")
    add_common_dataset_args(evaluate)
    add_common_env_args(evaluate)
    evaluate.add_argument("--checkpoint", default=None)
    evaluate.add_argument("--baseline", choices=("hold", "fixed", "random", "queue_greedy"), default=None)
    evaluate.add_argument("--policy-arch", choices=POLICY_ARCHES, default="single_head_with_district_feature")
    evaluate.add_argument("--fixed-green-time", type=int, default=20)
    evaluate.add_argument("--split", default="val", choices=("train", "val", "test"))
    evaluate.add_argument("--scenarios-per-city", type=int, default=1)
    evaluate.add_argument("--max-val-cities", type=int, default=None)
    evaluate.add_argument("--city-id", default=None)
    evaluate.add_argument("--scenario-name", default=None)
    evaluate.add_argument(
        "--eval-num-workers",
        type=int,
        default=1,
        help="CPU workers for evaluation. -1 uses all CPU cores.",
    )
    evaluate.add_argument("--device", default=None)

    return parser.parse_args()


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")


def add_common_env_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--reward-variant", choices=REWARD_VARIANTS, default="current")
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


def build_env_config(args: argparse.Namespace) -> EnvConfig:
    include_district_type_feature = getattr(args, "policy_arch", "multi_head") != "single_head"
    return EnvConfig(
        simulator_interval=args.simulator_interval,
        decision_interval=args.decision_interval,
        min_green_time=args.min_green_time,
        thread_num=args.thread_num,
        max_episode_seconds=args.max_episode_seconds,
        observation=ObservationConfig(
            max_incoming_lanes=args.max_incoming_lanes,
            count_scale=args.count_scale,
            elapsed_time_scale=args.elapsed_time_scale,
            include_outgoing_congestion=not args.disable_outgoing_congestion,
            include_district_context=not args.disable_district_context,
            include_district_type_feature=include_district_type_feature,
        ),
        reward=RewardConfig(
            variant=args.reward_variant,
            waiting_weight=args.waiting_weight,
            vehicle_weight=args.vehicle_weight,
            pressure_weight=args.pressure_weight,
            reward_scale=args.reward_scale,
            normalize_by_lane_count=not args.disable_lane_reward_normalization,
            clip_reward=args.reward_clip,
            queue_delta_weight=args.queue_delta_weight,
            wait_delta_weight=args.wait_delta_weight,
            queue_level_weight=args.queue_level_weight,
            wait_level_weight=args.wait_level_weight,
            throughput_weight=args.throughput_weight,
            imbalance_weight=args.imbalance_weight,
            delta_clip=args.reward_delta_clip,
            level_normalizer=args.reward_level_normalizer,
            throughput_normalizer=args.throughput_normalizer,
        ),
    )


def main() -> None:
    args = parse_args()
    if args.command == "make-splits":
        handle_make_splits(args)
        return
    if args.command == "train":
        handle_train(args)
        return
    if args.command == "evaluate":
        handle_evaluate(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


def handle_make_splits(args: argparse.Namespace) -> None:
    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    splits = dataset.generate_default_splits(
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        overwrite=args.overwrite,
    )
    print(json.dumps(splits, indent=2))


def handle_train(args: argparse.Namespace) -> None:
    from training.trainer import DQNConfig, DQNTrainer

    if args.train_scenario_name and not args.train_city_id:
        raise ValueError("--train-scenario-name requires --train-city-id.")

    if args.debug_fast:
        args.total_updates = min(args.total_updates, 10)
        args.max_train_cities = 2
        args.max_val_cities = 1
        args.val_scenarios_per_city = 1
        args.rollout_decision_steps = min(args.rollout_decision_steps, 64)
        args.eval_every = max(args.total_updates + 1, 100000)
        args.checkpoint_every = max(args.total_updates + 1, 100000)
        args.disable_baseline_comparison = True
        args.verbose_progress = True
        args.gradient_steps = min(args.gradient_steps, 8)
        args.learning_starts = min(args.learning_starts, 1000)
        args.replay_capacity = min(args.replay_capacity, 20000)

    if args.fast_overfit:
        args.gradient_steps = max(args.gradient_steps, 128)
        args.learning_starts = min(args.learning_starts, 2000)
        args.rollout_decision_steps = min(args.rollout_decision_steps, 128)
        args.eval_every = max(args.total_updates + 1, 100000)
        args.checkpoint_every = min(args.checkpoint_every, 25)
        args.disable_baseline_comparison = True
        args.thread_num = max(args.thread_num, 4)
        args.verbose_progress = True
        args.num_rollout_workers = max(args.num_rollout_workers, 4)
        if args.train_city_id is not None:
            args.max_train_cities = 1
        if args.overfit_val_on_train_scenario:
            args.max_val_cities = 1
            args.val_scenarios_per_city = 1

    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    dataset.generate_default_splits()

    env_config = build_env_config(args)
    dqn_config = DQNConfig(
        policy_arch=args.policy_arch,
        total_updates=args.total_updates,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        n_step=args.n_step,
        replay_capacity=args.replay_capacity,
        minibatch_size=args.minibatch_size,
        learning_starts=args.learning_starts,
        gradient_steps=args.gradient_steps,
        target_tau=args.target_tau,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dueling=not args.disable_dueling,
        seed=args.seed,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        val_scenarios_per_city=args.val_scenarios_per_city,
        max_val_cities=args.max_val_cities,
        max_train_cities=args.max_train_cities,
        num_rollout_workers=args.num_rollout_workers,
        rollout_episodes_per_update=args.rollout_episodes_per_update,
        train_city_id=args.train_city_id,
        train_scenario_name=args.train_scenario_name,
        overfit_val_on_train_scenario=args.overfit_val_on_train_scenario,
        rollout_decision_steps=args.rollout_decision_steps,
        resume_from=args.resume_from,
        use_observation_normalization=not args.disable_obs_norm,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        prioritized_replay_beta_start=args.prioritized_replay_beta_start,
        prioritized_replay_beta_end=args.prioritized_replay_beta_end,
        prioritized_replay_beta_steps=args.prioritized_replay_beta_steps,
        rolling_window_size=args.rolling_window_size,
        checkpoint_on_eval=not args.disable_checkpoint_on_eval,
        compare_baselines=not args.disable_baseline_comparison,
        skip_failed_validation_episodes=not args.fail_on_val_error,
        verbose_progress=args.verbose_progress,
        eval_num_workers=args.eval_num_workers,
        enable_tensorboard=not args.disable_tensorboard,
        tensorboard_log_dir=args.tensorboard_log_dir,
        use_tqdm=not args.disable_tqdm,
    )

    trainer = DQNTrainer(
        dataset=dataset,
        env_config=env_config,
        dqn_config=dqn_config,
        output_dir=args.output_dir,
        device=args.device,
    )
    trainer.fit()


def handle_evaluate(args: argparse.Namespace) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import torch

    from training.device import configure_torch_runtime, resolve_torch_device
    from training.models import RunningNormalizer, TrafficControlQNetwork
    from training.rollout import evaluate_policy
    from training.trainer import (
        _build_standalone_eval_context,
        _init_parallel_eval_worker_from_context,
        _parallel_eval_worker,
        aggregate_metrics,
        aggregate_metrics_by_scenario,
    )

    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    dataset.generate_default_splits()
    env_config = build_env_config(args)
    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    print(
        "[setup] "
        f"torch_device={device.type} "
        f"policy_arch={args.policy_arch} "
        f"reward_variant={env_config.reward.variant}"
    )

    if args.city_id and args.scenario_name:
        scenario_specs = [dataset.build_scenario_spec(args.city_id, args.scenario_name)]
    else:
        scenario_specs = dataset.iter_scenarios(
            split_name=args.split,
            scenarios_per_city=args.scenarios_per_city,
            max_cities=args.max_val_cities,
            diversify_single_scenario=True,
        )

    actor = None
    obs_normalizer = None
    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("env_config"):
            env_config = load_env_config(checkpoint["env_config"])
        network_architecture = checkpoint.get("network_architecture") or checkpoint.get("policy_architecture", {})
        trainer_config = checkpoint.get("dqn_config", {})
        checkpoint_policy_arch = network_architecture.get(
            "policy_arch",
            trainer_config.get("policy_arch", "single_head_with_district_feature"),
        )
        actor = TrafficControlQNetwork(
            observation_dim=int(
                network_architecture.get(
                    "observation_dim",
                    observation_dim_for_spec(env_config, scenario_specs[0]),
                )
            ),
            action_dim=int(network_architecture.get("action_dim", 2)),
            hidden_dim=int(trainer_config.get("hidden_dim", 256)),
            num_layers=int(trainer_config.get("hidden_layers", 2)),
            district_types=tuple(network_architecture.get("district_types", DISTRICT_TYPES)),
            policy_arch=checkpoint_policy_arch,
            dueling=bool(network_architecture.get("dueling", True)),
        ).to(device)
        actor.load_state_dict(
            checkpoint.get("q_network_state_dict") or checkpoint["policy_state_dict"]
        )
        actor.eval()
        if checkpoint.get("obs_normalizer"):
            obs_normalizer = RunningNormalizer()
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        print(
            "[setup] "
            f"checkpoint_policy_arch={checkpoint_policy_arch} "
            f"checkpoint_reward_variant={env_config.reward.variant}"
        )
    elif args.baseline == "hold":
        actor = HoldPhasePolicy()
    elif args.baseline == "fixed":
        actor = FixedCyclePolicy(green_time=args.fixed_green_time)
    elif args.baseline == "random":
        actor = RandomPhasePolicy(seed=7)
    elif args.baseline == "queue_greedy":
        actor = QueueGreedyPolicy()
    else:
        raise ValueError("Provide either --checkpoint or --baseline.")

    episode_metrics = []
    resolved_eval_workers = resolve_eval_workers(args.eval_num_workers, len(scenario_specs))
    if resolved_eval_workers > 1:
        worker_context = _build_standalone_eval_context(
            env_config=env_config,
            actor=actor,
            obs_normalizer=obs_normalizer,
            device=device,
            seed=7,
            fixed_green_time=args.fixed_green_time,
            baseline_name=args.baseline,
        )
        with ProcessPoolExecutor(
            max_workers=resolved_eval_workers,
            initializer=_init_parallel_eval_worker_from_context,
            initargs=(worker_context,),
        ) as executor:
            futures = {
                executor.submit(_parallel_eval_worker, scenario_spec, index, "standalone"): (
                    scenario_spec,
                    index,
                )
                for index, scenario_spec in enumerate(scenario_specs, start=1)
            }
            total_specs = len(scenario_specs)
            for future in as_completed(futures):
                scenario_spec, index = futures[future]
                metrics = future.result()
                episode_metrics.append(metrics)
                print(
                    f"[eval] city={scenario_spec.city_id} "
                    f"scenario={scenario_spec.scenario_name} i={index}/{total_specs}"
                )
                print(json.dumps(metrics))
    else:
        for scenario_spec in scenario_specs:
            metrics = evaluate_policy(
                env_factory=lambda spec=scenario_spec: build_env(env_config, spec),
                actor=actor,
                device=device,
                obs_normalizer=obs_normalizer,
                deterministic=True,
            )
            episode_metrics.append(metrics)
            print(json.dumps(metrics))

    aggregate = aggregate_metrics(episode_metrics)
    aggregate.update(aggregate_metrics_by_scenario(episode_metrics))
    print(json.dumps(aggregate, indent=2))


def build_env(env_config: EnvConfig, scenario_spec) -> object:
    from env.traffic_env import TrafficEnv

    return TrafficEnv(
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


def observation_dim_for_spec(env_config: EnvConfig, scenario_spec) -> int:
    env = build_env(env_config, scenario_spec)
    return env.observation_dim


def resolve_eval_workers(requested_workers: int, total_specs: int) -> int:
    if requested_workers == -1:
        requested_workers = os.cpu_count() or 1
    if requested_workers <= 1:
        return 1
    return min(requested_workers, total_specs)


def load_env_config(payload: dict) -> EnvConfig:
    return EnvConfig(
        simulator_interval=payload["simulator_interval"],
        decision_interval=payload["decision_interval"],
        min_green_time=payload["min_green_time"],
        thread_num=payload["thread_num"],
        max_episode_seconds=payload["max_episode_seconds"],
        observation=ObservationConfig(**payload["observation"]),
        reward=RewardConfig(**payload["reward"]),
    )


if __name__ == "__main__":
    main()
