"""
One-way ANOVA comparing the learned DQN policy against FixedCycle and Random
baselines across the same set of evaluation scenarios.

Output is always saved to --output-dir (default: results/anova/):
  - anova_report.txt   : human-readable results table
  - anova_results.json : raw per-episode data + full statistical results

Usage:
    python scripts/anova_test.py --checkpoint artifacts/dqn_shared/best_validation.pt
    python scripts/anova_test.py --checkpoint artifacts/dqn_shared/best_validation.pt \
        --split test --scenarios-per-city 3 --output-dir results/anova
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.local_policy import FixedCyclePolicy, RandomPhasePolicy
from training.dataset import CityFlowDataset
from training.device import configure_torch_runtime, resolve_torch_device
from training.models import RunningNormalizer, TrafficControlQNetwork
from training.rollout import evaluate_policy
from training.train_local_policy import build_env, build_env_config, load_env_config

# Metrics to run ANOVA on: (result_key, display_label, lower_is_better)
ANOVA_METRICS = [
    ("episode_return", "Episode Return", False),
    ("mean_waiting_vehicles", "Mean Waiting Vehicles", True),
    ("average_travel_time", "Average Travel Time (s)", True),
    ("throughput", "Throughput (vehicles)", False),
]

POLICY_NAMES = ("learned", "fixed", "random")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ANOVA test: RL vs baselines.")
    p.add_argument(
        "--checkpoint",
        default="artifacts/dqn_shared/best_validation.pt",
        help="Path to the trained DQN checkpoint.",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate on.",
    )
    p.add_argument("--scenarios-per-city", type=int, default=1)
    p.add_argument("--max-cities", type=int, default=None)
    p.add_argument("--generated-root", default="data/generated")
    p.add_argument("--splits-root", default="data/splits")
    p.add_argument("--device", default=None)
    p.add_argument("--fixed-green-time", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=7)
    p.add_argument(
        "--output-dir",
        default="results/anova",
        help="Directory where the text report and JSON results are saved (created if missing).",
    )
    p.add_argument("--disable-tqdm", action="store_true")
    # Env config args — defaults match training defaults
    p.add_argument("--decision-interval", type=int, default=5)
    p.add_argument("--simulator-interval", type=int, default=1)
    p.add_argument("--min-green-time", type=int, default=10)
    p.add_argument("--thread-num", type=int, default=1)
    p.add_argument("--max-episode-seconds", type=int, default=None)
    p.add_argument("--max-incoming-lanes", type=int, default=16)
    p.add_argument("--count-scale", type=float, default=20.0)
    p.add_argument("--elapsed-time-scale", type=float, default=60.0)
    p.add_argument("--disable-district-context", action="store_true")
    p.add_argument("--disable-outgoing-congestion", action="store_true")
    p.add_argument("--reward-variant", default="wait_queue_throughput")
    p.add_argument("--waiting-weight", type=float, default=1.0)
    p.add_argument("--vehicle-weight", type=float, default=0.1)
    p.add_argument("--pressure-weight", type=float, default=0.0)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--disable-lane-reward-normalization", action="store_true")
    p.add_argument("--reward-clip", type=float, default=5.0)
    p.add_argument("--queue-delta-weight", type=float, default=2.0)
    p.add_argument("--wait-delta-weight", type=float, default=4.0)
    p.add_argument("--queue-level-weight", type=float, default=0.5)
    p.add_argument("--wait-level-weight", type=float, default=1.0)
    p.add_argument("--throughput-weight", type=float, default=0.1)
    p.add_argument("--imbalance-weight", type=float, default=0.1)
    p.add_argument("--reward-delta-clip", type=float, default=2.0)
    p.add_argument("--reward-level-normalizer", type=float, default=10.0)
    p.add_argument("--throughput-normalizer", type=float, default=2.0)
    p.add_argument("--policy-arch", default="single_head_with_district_feature")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_episode_metrics(
    policies: dict,
    scenario_specs: list,
    disable_tqdm: bool,
) -> dict[str, list[dict]]:
    """Run each policy over all scenarios and return raw per-episode metric dicts."""
    all_metrics: dict[str, list[dict]] = {name: [] for name in policies}

    for name, (actor, device, normalizer, env_factory_fn) in policies.items():
        print(f"\n[collect] policy={name}  n_scenarios={len(scenario_specs)}")
        iterator = enumerate(scenario_specs, start=1)
        if not disable_tqdm:
            iterator = tqdm(
                iterator,
                total=len(scenario_specs),
                desc=f"anova:{name}",
                dynamic_ncols=True,
                leave=False,
            )
        for _idx, spec in iterator:
            m = evaluate_policy(
                env_factory=lambda s=spec, ef=env_factory_fn: ef(s),
                actor=actor,
                device=device,
                obs_normalizer=normalizer,
                deterministic=True,
            )
            all_metrics[name].append(m)
            if not disable_tqdm:
                iterator.set_postfix(
                    city=spec.city_id,
                    ret=f"{m['episode_return']:.3f}",
                )
    return all_metrics


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def extract_metric(episode_list: list[dict], key: str) -> np.ndarray:
    return np.array([ep[key] for ep in episode_list if key in ep], dtype=float)


def run_anova(groups: dict[str, np.ndarray]) -> dict:
    """One-way ANOVA with normality/variance checks, Kruskal-Wallis fallback, and Tukey HSD."""
    arrays = list(groups.values())
    names = list(groups.keys())

    # Shapiro-Wilk normality test per group
    normality: dict[str, dict] = {}
    all_normal = True
    for name, arr in zip(names, arrays):
        if len(arr) >= 3:
            stat, pval = stats.shapiro(arr)
            normality[name] = {"statistic": float(stat), "p_value": float(pval)}
            if pval < 0.05:
                all_normal = False
        else:
            normality[name] = {"statistic": None, "p_value": None}

    # Levene's test for homogeneity of variance
    levene_stat, levene_p = stats.levene(*arrays)
    equal_variance = levene_p >= 0.05

    # One-way ANOVA
    f_stat, anova_p = stats.f_oneway(*arrays)

    # Effect size: eta-squared (SS_between / SS_total)
    grand_mean = np.concatenate(arrays).mean()
    ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
    ss_total = sum(((a - grand_mean) ** 2).sum() for a in arrays)
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

    # Kruskal-Wallis (non-parametric; always computed for reference)
    kw_stat, kw_p = stats.kruskal(*arrays)

    # Tukey HSD pairwise (scipy >= 1.8)
    tukey_result = stats.tukey_hsd(*arrays)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pval = tukey_result.pvalue[i, j]
            pairs.append({
                "group_a": names[i],
                "group_b": names[j],
                "mean_a": float(arrays[i].mean()),
                "mean_b": float(arrays[j].mean()),
                "difference": float(arrays[i].mean() - arrays[j].mean()),
                "p_value": float(pval),
                "significant": bool(pval < 0.05),
            })

    return {
        "n_per_group": {n: int(len(a)) for n, a in zip(names, arrays)},
        "means": {n: float(a.mean()) for n, a in zip(names, arrays)},
        "stds": {n: float(a.std()) for n, a in zip(names, arrays)},
        "normality": normality,
        "all_normal": all_normal,
        "levene": {"statistic": float(levene_stat), "p_value": float(levene_p)},
        "equal_variance": equal_variance,
        "anova": {"f_statistic": float(f_stat), "p_value": float(anova_p)},
        "kruskal_wallis": {"h_statistic": float(kw_stat), "p_value": float(kw_p)},
        "eta_squared": eta_squared,
        "tukey_hsd": pairs,
        "recommended_test": "ANOVA" if all_normal and equal_variance else "Kruskal-Wallis",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_results(results: dict[str, dict]) -> str:
    lines: list[str] = []
    sep = "=" * 78
    thin = "-" * 78

    lines.append("")
    lines.append(sep)
    lines.append("ANOVA RESULTS: Learned DQN vs FixedCycle vs Random")
    lines.append(sep)

    for metric_key, label, lower_is_better in ANOVA_METRICS:
        if metric_key not in results:
            continue
        r = results[metric_key]
        test = r["recommended_test"]
        if test == "ANOVA":
            stat_val = r["anova"]["f_statistic"]
            p_val = r["anova"]["p_value"]
            stat_label = "F"
        else:
            stat_val = r["kruskal_wallis"]["h_statistic"]
            p_val = r["kruskal_wallis"]["p_value"]
            stat_label = "H"

        lines.append("")
        lines.append(thin)
        lines.append(f"  Metric : {label}")
        lines.append(
            f"  Test   : {test}  "
            f"({stat_label}={stat_val:.4f},  p={p_val:.4f} {sig_stars(p_val)},  "
            f"eta2={r['eta_squared']:.4f})"
        )
        lines.append(f"  n      : {r['n_per_group']}")
        lines.append("  Means  :")
        for name in POLICY_NAMES:
            if name not in r["means"]:
                continue
            direction = "(higher=better)" if not lower_is_better else "(lower=better)"
            suffix = f"  {direction}" if name == "learned" else ""
            lines.append(
                f"           {name:10s}  {r['means'][name]:10.4f}  +/-  {r['stds'][name]:.4f}{suffix}"
            )

        lines.append("  Tukey HSD pairwise:")
        for pair in r["tukey_hsd"]:
            sig_label = "SIGNIFICANT" if pair["significant"] else "not significant"
            rl_note = ""
            if "learned" in (pair["group_a"], pair["group_b"]):
                ga, gb = pair["group_a"], pair["group_b"]
                diff = pair["difference"]
                learned_better = (
                    (ga == "learned" and not lower_is_better and diff > 0)
                    or (ga == "learned" and lower_is_better and diff < 0)
                    or (gb == "learned" and not lower_is_better and diff < 0)
                    or (gb == "learned" and lower_is_better and diff > 0)
                )
                rl_note = "  [RL wins]" if learned_better else "  [RL loses]"
            lines.append(
                f"           {pair['group_a']:10s} vs {pair['group_b']:10s}  "
                f"delta={pair['difference']:+.4f}  "
                f"p={pair['p_value']:.4f} {sig_stars(pair['p_value'])}  "
                f"{sig_label}{rl_note}"
            )

        if not r["all_normal"]:
            lines.append("  [!] Normality violated for at least one group -- Kruskal-Wallis preferred.")
        if not r["equal_variance"]:
            lines.append("  [!] Levene's test failed (unequal variances).")

    lines.append("")
    lines.append(sep)
    lines.append("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    lines.append(sep)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "anova_report.txt"
    json_path = output_dir / "anova_results.json"

    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    scenario_specs = dataset.iter_scenarios(
        split_name=args.split,
        scenarios_per_city=args.scenarios_per_city,
        max_cities=args.max_cities,
        diversify_single_scenario=True,
    )
    print(f"[setup] split={args.split}  n_scenarios={len(scenario_specs)}")
    print(f"[setup] output_dir={output_dir.resolve()}")

    if len(scenario_specs) < 3:
        print(
            f"WARNING: Only {len(scenario_specs)} scenario(s) found. ANOVA requires independent "
            "observations per group. Use --scenarios-per-city or a larger split for reliable results."
        )

    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    print(f"[setup] torch_device={device.type}")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    env_config = build_env_config(args)
    if checkpoint.get("env_config"):
        env_config = load_env_config(checkpoint["env_config"])
        print("[setup] env_config loaded from checkpoint")

    network_architecture = checkpoint.get("network_architecture") or checkpoint.get(
        "policy_architecture", {}
    )
    trainer_config = checkpoint.get("dqn_config", {})
    policy_arch = network_architecture.get(
        "policy_arch", trainer_config.get("policy_arch", args.policy_arch)
    )

    dqn = TrafficControlQNetwork(
        observation_dim=int(network_architecture["observation_dim"]),
        action_dim=int(network_architecture.get("action_dim", 2)),
        hidden_dim=int(trainer_config.get("hidden_dim", 256)),
        num_layers=int(trainer_config.get("hidden_layers", 2)),
        district_types=tuple(network_architecture.get("district_types", ())),
        policy_arch=policy_arch,
        dueling=bool(network_architecture.get("dueling", True)),
    ).to(device)
    dqn.load_state_dict(
        checkpoint.get("q_network_state_dict") or checkpoint["policy_state_dict"]
    )
    dqn.eval()

    obs_normalizer = None
    if checkpoint.get("obs_normalizer"):
        obs_normalizer = RunningNormalizer()
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

    print(f"[setup] checkpoint={checkpoint_path.name}  policy_arch={policy_arch}")

    def env_factory(spec):
        return build_env(env_config, spec)

    policies = {
        "learned": (dqn, device, obs_normalizer, env_factory),
        "fixed": (FixedCyclePolicy(green_time=args.fixed_green_time), None, None, env_factory),
        "random": (RandomPhasePolicy(seed=args.random_seed), None, None, env_factory),
    }

    # --- Collect per-episode raw data ---
    raw_data = collect_episode_metrics(policies, scenario_specs, args.disable_tqdm)

    # --- Run ANOVA for each metric ---
    anova_results: dict[str, dict] = {}
    for metric_key, _label, _lower in ANOVA_METRICS:
        groups = {
            name: extract_metric(episodes, metric_key)
            for name, episodes in raw_data.items()
        }
        if any(len(arr) == 0 for arr in groups.values()):
            print(f"[anova] skipping {metric_key} -- not present in all policy outputs")
            continue
        anova_results[metric_key] = run_anova(groups)

    # --- Format and save report ---
    report_text = format_results(anova_results)
    print(report_text)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"[output] report saved to {report_path}")

    # --- Save JSON ---
    payload = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "n_scenarios": len(scenario_specs),
        "raw_episode_data": raw_data,
        "anova_results": anova_results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[output] JSON saved to {json_path}")


if __name__ == "__main__":
    main()
