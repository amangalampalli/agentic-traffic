# training

Training, evaluation, device selection, and dataset utilities for the local multi-agent DQN stack.

## Main files

- [train_local_policy.py](/Users/aditya/Developer/traffic-llm/training/train_local_policy.py)
  Main CLI for split generation, DQN training, and checkpoint evaluation.
- [trainer.py](/Users/aditya/Developer/traffic-llm/training/trainer.py)
  Replay buffer, DQN training loop, checkpointing, validation, and aggregate metrics.
- [rollout.py](/Users/aditya/Developer/traffic-llm/training/rollout.py)
  Evaluation helper shared by learned agents and rule-based baselines.
- [models.py](/Users/aditya/Developer/traffic-llm/training/models.py)
  Dueling Q-network and running observation normalization.
- [device.py](/Users/aditya/Developer/traffic-llm/training/device.py)
  Torch device selection for `cuda`, `mps`, or `cpu`.
- [dataset.py](/Users/aditya/Developer/traffic-llm/training/dataset.py)
  City discovery, split loading, and scenario sampling.

## Main entry points

- `python3 -m training.train_local_policy make-splits`
- `python3 -m training.train_local_policy train`
- `python3 -m training.train_local_policy train --max-val-cities 3 --val-scenarios-per-city 7`
- `python3 -m training.train_local_policy train --max-train-cities 70 --max-val-cities 3 --val-scenarios-per-city 7 --policy-arch single_head_with_district_feature --reward-variant wait_queue_throughput`
- `python3 -m training.train_local_policy train --policy-arch single_head_with_district_feature --reward-variant wait_queue_throughput`
- `python3 -m training.train_local_policy evaluate --checkpoint artifacts/dqn_shared/best_validation.pt --split val`
- `python3 -m training.train_local_policy evaluate --baseline queue_greedy --split val`
- `tensorboard --logdir artifacts/dqn_shared/tensorboard`

## Training flow

1. Load city-level splits from `data/splits/`.
2. Sample one `(city, scenario)` episode at a time from the train split.
3. Run one shared Q-network across all controlled intersections in that city.
4. Collect per-intersection transitions into prioritized replay with n-step returns, using parallel CPU rollout workers by default.
5. Perform Double DQN updates against a target network.
6. Periodically evaluate on validation cities and save checkpoints. By default, eval and checkpoint cadence are both 40 updates, and each validation pass also writes an `update_XXXX.pt` checkpoint.
