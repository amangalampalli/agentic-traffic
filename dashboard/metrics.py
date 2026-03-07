from __future__ import annotations

from typing import Any


def extract_step_metrics(info: dict[str, Any]) -> dict[str, float]:
    metrics = info.get("metrics", {})
    return {
        "total_waiting": float(metrics.get("total_waiting", 0.0)),
        "total_queue": float(metrics.get("total_queue", 0.0)),
        "mean_reward": float(metrics.get("mean_reward", 0.0)),
        "num_intersections": float(metrics.get("num_intersections", 0.0)),
    }


def summarize_history(history: list[dict[str, Any]]) -> dict[str, float]:
    if not history:
        return {
            "avg_total_waiting": 0.0,
            "avg_total_queue": 0.0,
            "avg_mean_reward": 0.0,
            "num_steps": 0.0,
        }

    total_waiting = 0.0
    total_queue = 0.0
    mean_reward = 0.0

    for row in history:
        metrics = row.get("metrics", {})
        total_waiting += float(metrics.get("total_waiting", 0.0))
        total_queue += float(metrics.get("total_queue", 0.0))
        mean_reward += float(metrics.get("mean_reward", 0.0))

    n = len(history)
    return {
        "avg_total_waiting": total_waiting / n,
        "avg_total_queue": total_queue / n,
        "avg_mean_reward": mean_reward / n,
        "num_steps": float(n),
    }


def flatten_directives(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for row in history:
        step = row.get("step", 0)
        directives = row.get("district_directives", {})
        for district_id, directive in directives.items():
            rows.append(
                {
                    "step": step,
                    "district_id": district_id,
                    "mode": directive.get("mode", "none"),
                    "duration": directive.get("duration", 1),
                    "district_weight": directive.get("district_weight", 0.5),
                    "corridor": directive.get("corridor"),
                    "rationale": directive.get("rationale", ""),
                }
            )

    return rows
