# Final Ablation Runbook

## Dataset

Generate the constrained v3 dataset:

```bash
python scripts/generate_large_district_dataset.py \
  --num-train 10000 \
  --num-val 2500 \
  --output-dir data/district_llm_dataset_v3 \
  --checkpoint artifacts/dqn_shared/best_validation.pt \
  --max-candidate-intersections 6 \
  --max-target-intersections 3
```

Defaults:

- candidate pool is visible in the prompt via `candidate_intersections`
- labels are constrained to visible candidates
- DQN teacher sources are preferred by default

## Notebook

Use [notebooks/llama_finetune.ipynb](/root/aditya/agentic-traffic/notebooks/llama_finetune.ipynb).

Recommended defaults for the A100 main run:

- `RUN_MODE = "main_run"`
- `num_train_epochs = 2`
- `per_device_train_batch_size = 8`
- `gradient_accumulation_steps = 4`
- effective batch size = 32
- `learning_rate = 1e-4`
- `warmup_ratio = 0.05`
- `eval_steps = 100`
- `save_steps = 100`

Smoke test mode:

- `RUN_MODE = "smoke_test"`
- short `max_steps`
- verifies formatting, checkpointing, and eval wiring

Optional max-step override:

- set `MAX_STEPS_OVERRIDE = 5000` only for explicit experimentation
- do not use it as the default main run

Artifacts:

- checkpoints: `artifacts/district_llm_adapter_v3/<run_mode>/checkpoints`
- saved adapter: `artifacts/district_llm_adapter_v3/<run_mode>/adapter`

## Evaluation

Run offline eval with repair enabled:

```bash
python -m district_llm.eval \
  --model-path artifacts/district_llm_adapter_v3/main_run/adapter \
  --val-jsonl data/district_llm_dataset_v3/val.jsonl \
  --generated-root data/generated \
  --max-examples 250 \
  --debug-examples 10 \
  --allow-only-visible-candidates \
  --max-target-intersections 3 \
  --fallback-on-empty-targets \
  --fallback-mode heuristic \
  --restrict-targets-to-visible-summary \
  --report-before-after-repair
```

Key outputs:

- raw vs repaired target metrics
- invalid target-id rate before and after repair
- visible-candidate-restricted metrics
- target failure buckets and debug examples
