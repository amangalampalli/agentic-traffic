---
title: Agentic Traffic
emoji: 🏢
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
short_description: Agentic AI to control traffic lights
app_port: 7860
---

# traffic-llm

CityFlow-based traffic-control project with intersection-level multi-agent DQN training and district-aware policy variants.

Full model weights and files can be found here: https://huggingface.co/Aditya2162/agentic-traffic

## OpenEnv UI

For the deployed OpenEnv web interface:

- Click `Reset` before using `Step`.
- Leave `Use Llm` unchecked for the fast, stable DQN-only path.
- Use `District Actions` = `{}` for a valid no-op step payload.
- Only enable `Use Llm` when you explicitly want district-level LLM guidance on top of the DQN executor.

## Training

The default local-policy trainer now uses parameter-shared dueling Double DQN with prioritized replay and n-step returns:

```bash
python3 -m training.train_local_policy train
```

That trains against `data/generated`, uses `data/splits`, writes checkpoints to `artifacts/dqn_shared`, enables TensorBoard logging, uses parallel CPU rollout workers by default, shows `tqdm` progress bars, and now validates plus checkpoints every 40 updates by default.

For a broader but still manageable validation pass:

```bash
python3 -m training.train_local_policy train --max-val-cities 3 --val-scenarios-per-city 7
```

That evaluates 3 validation cities across all 7 scenario types. This gives 21 learned-policy validation episodes per eval, or 63 total episodes if random and fixed baselines are also enabled.

Phase-3-style full training with the same 40-update eval/checkpoint cadence:

```bash
python3 -m training.train_local_policy train \
  --max-train-cities 70 \
  --max-val-cities 3 \
  --val-scenarios-per-city 7 \
  --policy-arch single_head_with_district_feature \
  --reward-variant wait_queue_throughput
```

Useful ablations:

```bash
python3 -m training.train_local_policy train --policy-arch multi_head --reward-variant current
python3 -m training.train_local_policy train --policy-arch single_head --reward-variant current
python3 -m training.train_local_policy train --policy-arch single_head_with_district_feature --reward-variant wait_queue_throughput
```

For a fast phase-1 overfit run on one fixed world:

```bash
python3 -m training.train_local_policy train \
  --total-updates 25 \
  --train-city-id city_0072 \
  --train-scenario-name normal \
  --overfit-val-on-train-scenario \
  --fast-overfit \
  --policy-arch single_head_with_district_feature \
  --reward-variant wait_queue_throughput
```

To create or refresh dataset splits:

```bash
python3 -m training.train_local_policy make-splits
```

To evaluate the best checkpoint:

```bash
python3 -m training.train_local_policy evaluate \
  --checkpoint artifacts/dqn_shared/best_validation.pt \
  --split val
```

To evaluate a heuristic baseline directly:

```bash
python3 -m training.train_local_policy evaluate --baseline queue_greedy --split val
```

## TensorBoard

TensorBoard logs are written to `artifacts/dqn_shared/tensorboard` by default.

```bash
tensorboard --logdir artifacts/dqn_shared/tensorboard
```

## District LLM

The district LLM stack lives under `district_llm/`. It treats the learned DQN local controller as the low-level executor, derives district-scale SFT labels automatically from DQN rollout windows, and defaults district-model fine-tuning to DQN-derived rows only.

Generate district-LLM data from a learned checkpoint:

```bash
python3 -m district_llm.generate_dataset \
  --controller rl_checkpoint \
  --checkpoint artifacts/dqn_shared/best_validation.pt \
  --episodes 100 \
  --decision-interval 10 \
  --use-checkpoint-env-config \
  --output data/district_llm_train.jsonl
```

Generate from fixed or heuristic baselines:

```bash
python3 -m district_llm.generate_dataset --controller fixed --episodes 50 --decision-interval 10 --output data/district_llm_fixed.jsonl
python3 -m district_llm.generate_dataset --controller queue_greedy --episodes 50 --decision-interval 10 --output data/district_llm_heuristic.jsonl
python3 -m district_llm.generate_dataset --teacher-spec fixed --teacher-spec random --episodes 50 --decision-interval 10 --output data/district_llm_multi_teacher.jsonl
```

Train a first-pass district model with Unsloth/QLoRA:

```bash
python3 -m training.train_district_llm \
  --dataset data/district_llm_train.jsonl \
  --output-dir artifacts/district_llm_qwen \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --load-in-4bit \
  --lora-rank 16 \
  --max-seq-length 1024 \
  --max-steps 1000
```

Run single-sample inference:

```bash
python3 -m district_llm.inference \
  --model artifacts/district_llm_qwen \
  --city-id city_0006 \
  --scenario-name accident \
  --district-id d_00
```

Run the OpenEnv-compatible district wrapper on top of the current DQN stack:

```bash
uvicorn openenv_app.app:app --reload
```

## Algorithm

- Training algorithm: parameter-shared dueling Double DQN.
- Replay: prioritized replay over per-intersection transitions gathered from full CityFlow worlds.
- Return target: n-step bootstrap target with target-network updates.
- Execution: all controllable intersections act simultaneously every RL decision interval.
- Action space: `0 = hold current phase`, `1 = switch to next green phase`.
- Safety: `min_green_time` is enforced in the environment and exposed through action masking.

Policy architecture modes:

- `multi_head`: shared trunk with district-type-specific Q heads.
- `single_head`: one shared Q head for all intersections, with district type removed from the observation.
- `single_head_with_district_feature`: one shared Q head for all intersections, with district type left in the observation as an explicit feature.

Reward variants:

- `current`: backward-compatible waiting and queue penalty.
- `normalized_wait_queue`: normalized queue and waiting reduction reward.
- `wait_queue_throughput`: normalized queue/wait reduction plus throughput bonus and imbalance penalty.

## Smoke Test

To sanity-check one generated scenario with the real CityFlow environment:

```bash
python3 scripts/smoke_test_env.py --city-id city_0001 --scenario-name normal --policy random
```

## Project layout

- `agents/`: heuristic local policies and simple baselines.
- `env/`: CityFlow environment, topology parsing, observation building, and reward logic.
- `training/`: dataset utilities, replay-based DQN training, evaluation helpers, TensorBoard logging, and CLIs.
- `data/`: generated synthetic cities, split files, and dataset generation utilities.
- `scripts/`: utility scripts, including the CityFlow smoke test.
- `third_party/`: vendored dependencies, including CityFlow source.

## Notes

- The generated dataset is assumed to already exist under `data/generated`.
- District membership comes from `district_map.json`.
- District types come from `metadata.json`.
- Runtime training and evaluation require the `cityflow` Python module to be installed in the active environment.
