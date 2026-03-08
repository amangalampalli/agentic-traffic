# RL Guidance Eval Runbook

This ablation keeps the RL checkpoint fixed.

District guidance is only used at inference time through the wrapper in
`district_llm/rl_guidance_wrapper.py`. The safest default is
`target_only_soft`, which applies a small local Q-value bias only at
`target_intersections`.

## Wrapper Modes

- `no_op`: guidance is computed and logged, but RL actions are unchanged.
- `target_only_soft`: weak local prior on target intersections. Default debug mode.
- `target_only_medium`: same scope, slightly stronger.
- `corridor_soft`: small corridor prior on targets plus a few aligned boundary intersections.
- `global_soft`: weak district-wide prior. Use only as an ablation.
- `current_legacy`: reference mode approximating the old strong/global wrapper.

## Fast Debug Matrix

Use a short horizon first so the wrapper can be debugged quickly:

```bash
python scripts/eval_rl_guidance_ablation.py \
  --rl-checkpoint artifacts/dqn_shared/best_validation.pt \
  --llm-model-path artifacts/district_llm_adapter_v3/main_run/adapter \
  --modes rl_only rl_heuristic rl_llm \
  --wrapper-modes no_op target_only_soft current_legacy \
  --split val \
  --cities city_0001 \
  --scenarios normal \
  --seeds 7 11 13 \
  --num-episodes 1 \
  --max-episode-seconds 300 \
  --guidance-refresh-steps 10 \
  --guidance-persistence-steps 3 \
  --bias-strength 0.12 \
  --target-only-bias-strength 0.18 \
  --corridor-bias-strength 0.05 \
  --max-intersections-affected 3 \
  --fallback-policy hold_previous \
  --save-guidance-traces \
  --output-dir artifacts/rl_guidance_eval/debug_matrix_300s
```

This expands into the paired comparison:

- `rl_only`
- `rl_heuristic+no_op`
- `rl_heuristic+target_only_soft`
- `rl_heuristic+current_legacy`
- `rl_llm+no_op`
- `rl_llm+target_only_soft`
- `rl_llm+current_legacy`

That command runs a superset of the exact smaller matrix from the wrapper audit prompt. Focus analysis on:

- `rl_only`
- `rl_heuristic+no_op`
- `rl_heuristic+target_only_soft`
- `rl_llm+no_op`
- `rl_llm+target_only_soft`
- `rl_llm+current_legacy`

## What To Look At

Primary files:

- `summary.json`
- `episode_metrics.csv`
- `guidance_traces.jsonl`
- `config.json`

Key wrapper metrics in `episode_metrics.csv`:

- `wrapper_mode`
- `mean_bias_magnitude`
- `max_bias_magnitude`
- `avg_num_targeted_intersections`
- `avg_num_affected_intersections`
- `percent_steps_with_active_guidance`
- `num_guidance_refreshes`
- `num_noop_guidance_events`
- `fallback_policy_used_count`

Interpretation:

- If `rl_heuristic+no_op` and `rl_llm+no_op` match `rl_only`, the harness itself is fine.
- If `current_legacy` collapses while `target_only_soft` stays near `rl_only`, the wrapper was too strong/global.
- If `rl_llm+target_only_soft` diverges from `rl_heuristic+target_only_soft`, the LLM is adding signal under safe integration.
- If `avg_num_affected_intersections` is large or `percent_steps_with_active_guidance` is near `1.0`, the wrapper is still too persistent or too broad.
- If `fallback_policy_used_count` stays high in `rl_llm`, inspect `guidance_traces.jsonl` before trusting traffic metrics.

## Cheap Follow-Up Ablations

Softer local prior:

```bash
--wrapper-modes no_op target_only_soft target_only_medium
```

Scope ablation:

```bash
--wrapper-modes target_only_soft corridor_soft global_soft current_legacy
```

More conservative persistence:

```bash
--guidance-refresh-steps 8 --guidance-persistence-steps 2
```

## Output Layout

Outputs are saved under the requested directory, for example:

```text
artifacts/rl_guidance_eval/debug_matrix_300s/
  config.json
  summary.json
  episode_metrics.csv
  episode_metrics.jsonl
  guidance_traces.jsonl
  seeded_configs/
```
