from __future__ import annotations

from typing import Any


class HeuristicController:
    """
    Simple local traffic-light controller.

    Action space:
        0 -> choose NS green
        1 -> choose EW green

    Assumes:
        queue_lengths = [N, S, E, W]
        waiting_counts = [N, S, E, W]
    """

    def __init__(
        self,
        min_green_steps: int = 5,
        switch_margin: float = 1.0,
        district_bonus_scale: float = 3.0,
        neighbor_pressure_scale: float = 0.25,
    ):
        self.min_green_steps = min_green_steps
        self.switch_margin = switch_margin
        self.district_bonus_scale = district_bonus_scale
        self.neighbor_pressure_scale = neighbor_pressure_scale

    def act(self, obs: dict[str, Any]) -> int:
        queue_lengths = obs.get("queue_lengths", [0, 0, 0, 0])
        waiting_counts = obs.get("waiting_counts", [0, 0, 0, 0])
        current_phase = int(obs.get("current_phase", 0))
        time_since_switch = int(obs.get("time_since_switch", 0))
        district_mode = obs.get("district_mode", "none")
        district_weight = float(obs.get("district_weight", 0.5))
        neighbor_pressure = obs.get("neighbor_pressure", [0.0, 0.0])

        ns_score = (
            queue_lengths[0]
            + queue_lengths[1]
            + 1.5 * (waiting_counts[0] + waiting_counts[1])
        )
        ew_score = (
            queue_lengths[2]
            + queue_lengths[3]
            + 1.5 * (waiting_counts[2] + waiting_counts[3])
        )

        # Optional small neighbor-pressure bias
        if isinstance(neighbor_pressure, list) and len(neighbor_pressure) >= 2:
            ns_score += self.neighbor_pressure_scale * float(neighbor_pressure[0])
            ew_score += self.neighbor_pressure_scale * float(neighbor_pressure[1])

        # District-level strategic bias
        district_bonus = self.district_bonus_scale * district_weight
        if district_mode == "prioritize_ns":
            ns_score += district_bonus
        elif district_mode == "prioritize_ew":
            ew_score += district_bonus
        elif district_mode == "green_wave":
            corridor = obs.get("district_corridor")
            if corridor == "ns":
                ns_score += district_bonus
            elif corridor == "ew":
                ew_score += district_bonus
        elif district_mode == "emergency_route":
            corridor = obs.get("district_corridor")
            if corridor in {"north_to_south", "south_to_north", "ns"}:
                ns_score += district_bonus * 1.5
            elif corridor in {"west_to_east", "east_to_west", "ew"}:
                ew_score += district_bonus * 1.5

        desired_phase = 0 if ns_score >= ew_score else 1

        # Avoid thrashing
        if time_since_switch < self.min_green_steps:
            return current_phase

        # Only switch if the other phase is meaningfully better
        current_score = ns_score if current_phase == 0 else ew_score
        desired_score = ns_score if desired_phase == 0 else ew_score

        if (
            desired_phase != current_phase
            and desired_score < current_score + self.switch_margin
        ):
            return current_phase

        return desired_phase
