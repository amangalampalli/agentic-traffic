# RL+LLM Wrapper Sweep

This sweep keeps both checkpoints fixed:

- RL weights stay fixed.
- LLM weights stay fixed.
- Only the inference-time `target_only_soft` wrapper settings change.

## Recommended First Sweep

Run the default cheap preset on one city, one scenario, and three seeds:

```bash
python scripts/sweep_rl_llm_wrapper.py \
  --rl-checkpoint artifacts/dqn_shared/best_validation.pt \
  --llm-model-path artifacts/district_llm_adapter_v3/main_run/adapter \
  --preset strength_targets_gating \
  --split val \
  --cities city_0001 \
  --scenarios normal \
  --seeds 7 11 13 \
  --episodes-per-seed 1 \
  --max-episode-seconds 300 \
  --guidance-refresh-steps 10 \
  --queue-threshold 150 \
  --imbalance-threshold 20 \
  --fallback-policy no_op \
  --output-dir artifacts/rl_llm_wrapper_sweep/first_pass
```

This preset sweeps a small curated grid over:

- `bias_strength` in `{0.025, 0.05, 0.075}`
- `max_intersections_affected` in `{1, 2}`
- `gating_mode` in `{always_on, incident_or_spillback, queue_or_imbalance}`
- `guidance_persistence_steps = 5`
- `enable_bias_decay = false`

It also includes `baseline_current_soft` as a reference row.

## Cheaper Probe

If you only want the fastest possible read on strength sensitivity:

```bash
python scripts/sweep_rl_llm_wrapper.py \
  --rl-checkpoint artifacts/dqn_shared/best_validation.pt \
  --llm-model-path artifacts/district_llm_adapter_v3/main_run/adapter \
  --preset strength_only \
  --cities city_0001 \
  --scenarios normal \
  --seeds 7 11 13 \
  --episodes-per-seed 1 \
  --max-episode-seconds 300 \
  --output-dir artifacts/rl_llm_wrapper_sweep/strength_only
```

## Broader Conservative Follow-Up

After the first pass identifies a promising strength/gating region:

```bash
python scripts/sweep_rl_llm_wrapper.py \
  --rl-checkpoint artifacts/dqn_shared/best_validation.pt \
  --llm-model-path artifacts/district_llm_adapter_v3/main_run/adapter \
  --preset full_conservative \
  --cities city_0001 \
  --scenarios normal \
  --seeds 7 11 13 \
  --episodes-per-seed 1 \
  --max-episode-seconds 300 \
  --output-dir artifacts/rl_llm_wrapper_sweep/full_conservative
```

## Outputs

Each sweep writes:

- `config.json`
- `sweep_results.csv`
- `sweep_results.parquet` when parquet support is available
- `paired_episode_metrics.csv`
- `ranking.json`
- `summary_report.json`
- optional `step_metrics.*`
- optional `guidance_traces.jsonl`

## What To Inspect

Start with:

- `summary_report.json`
- `ranking.json`
- `paired_episode_metrics.csv`

Key fields:

- `mean_return_delta_vs_rl_only`
- `mean_throughput_delta_vs_rl_only`
- `mean_avg_queue_delta_vs_rl_only`
- `mean_avg_wait_delta_vs_rl_only`
- `mean_percent_steps_with_active_guidance`
- `mean_avg_num_affected_intersections`
- `mean_num_steps_guidance_blocked_by_gate`

## Interpretation

The most promising configs should usually look like:

- small negative or positive `mean_return_delta_vs_rl_only`
- low `mean_avg_num_affected_intersections`
- moderate or low `mean_percent_steps_with_active_guidance`
- low fallback / invalid guidance counts

If the best configs cluster around:

- lower `bias_strength`
- `max_intersections_affected = 1`
- gated modes like `incident_or_spillback` or `queue_or_imbalance`

then the wrapper was still too active and guidance needs to remain a rare local prior.
