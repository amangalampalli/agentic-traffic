from __future__ import annotations

import json

from agents.district_coordinator import RuleBasedDistrictCoordinator
from agents.local_policy import SharedHeuristicLocalPolicy
from training.trainer import DistrictCoordinatorEvaluator


def make_env():
    from env.traffic_env import TrafficEnv
    from env.intersection_config import IntersectionConfig, DistrictConfig

    intersections = {
        "I1": IntersectionConfig(
            intersection_id="I1",
            district_id="D0",
            incoming_lanes=["I1_N", "I1_S", "I1_E", "I1_W"],
            outgoing_lanes=[],
            neighbors=["I2"],
            is_border=False,
        ),
        "I2": IntersectionConfig(
            intersection_id="I2",
            district_id="D0",
            incoming_lanes=["I2_N", "I2_S", "I2_E", "I2_W"],
            outgoing_lanes=[],
            neighbors=["I1", "I3"],
            is_border=True,
        ),
        "I3": IntersectionConfig(
            intersection_id="I3",
            district_id="D1",
            incoming_lanes=["I3_N", "I3_S", "I3_E", "I3_W"],
            outgoing_lanes=[],
            neighbors=["I2", "I4"],
            is_border=True,
        ),
        "I4": IntersectionConfig(
            intersection_id="I4",
            district_id="D1",
            incoming_lanes=["I4_N", "I4_S", "I4_E", "I4_W"],
            outgoing_lanes=[],
            neighbors=["I3"],
            is_border=False,
        ),
    }

    districts = {
        "D0": DistrictConfig(
            district_id="D0",
            intersection_ids=["I1", "I2"],
            neighbor_districts=["D1"],
        ),
        "D1": DistrictConfig(
            district_id="D1",
            intersection_ids=["I3", "I4"],
            neighbor_districts=["D0"],
        ),
    }

    return TrafficEnv(
        config_path="data/cityflow/config.json",
        intersections=intersections,
        districts=districts,
        coordination_interval=20,
        max_steps=200,
    )


def main():
    local_policy = SharedHeuristicLocalPolicy()
    evaluator = DistrictCoordinatorEvaluator(
        env_factory=make_env,
        local_policy=local_policy,
    )

    local_only = {}
    coordinated = {
        "D0": RuleBasedDistrictCoordinator(),
        "D1": RuleBasedDistrictCoordinator(),
    }

    results = evaluator.compare(
        seeds=[0, 1, 2, 3, 4],
        local_only_coordinators=local_only,
        coordinated_coordinators=coordinated,
        max_steps=200,
    )

    print(
        json.dumps(
            {
                "local_only": {
                    "avg_mean_reward": results["local_only"]["avg_mean_reward"],
                    "avg_total_waiting": results["local_only"]["avg_total_waiting"],
                    "avg_total_queue": results["local_only"]["avg_total_queue"],
                },
                "coordinated": {
                    "avg_mean_reward": results["coordinated"]["avg_mean_reward"],
                    "avg_total_waiting": results["coordinated"]["avg_total_waiting"],
                    "avg_total_queue": results["coordinated"]["avg_total_queue"],
                },
                "improvements": results["improvements"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
