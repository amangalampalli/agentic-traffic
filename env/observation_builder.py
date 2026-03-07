from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.intersection_config import DistrictConfig, IntersectionConfig
from env.utils import normalize_scalar


@dataclass(frozen=True)
class ObservationConfig:
    max_incoming_lanes: int = 16
    count_scale: float = 20.0
    elapsed_time_scale: float = 60.0
    include_outgoing_congestion: bool = True
    include_district_context: bool = True
    include_district_type_feature: bool = True


class ObservationBuilder:
    def __init__(
        self,
        intersections: dict[str, IntersectionConfig],
        districts: dict[str, DistrictConfig],
        config: ObservationConfig | None = None,
    ):
        self.intersections = intersections
        self.districts = districts
        self.config = config or ObservationConfig()
        self.intersection_ids = tuple(sorted(intersections))
        self._district_lookup = {
            intersection_id: intersections[intersection_id].district_id
            for intersection_id in self.intersection_ids
        }
        self._district_sizes = {
            district_id: max(1, len(district.intersection_ids))
            for district_id, district in districts.items()
        }
        self.observation_dim = self._compute_observation_dim()

    def build(
        self,
        lane_vehicle_count: dict[str, int],
        lane_waiting_count: dict[str, int],
        phase_positions: dict[str, int],
        phase_elapsed_times: dict[str, int],
        switch_allowed: dict[str, bool],
    ) -> dict[str, np.ndarray | tuple[str, ...]]:
        district_context = self._compute_district_context(
            lane_vehicle_count=lane_vehicle_count,
            lane_waiting_count=lane_waiting_count,
        )

        num_intersections = len(self.intersection_ids)
        max_lanes = self.config.max_incoming_lanes

        observations = np.zeros(
            (num_intersections, self.observation_dim),
            dtype=np.float32,
        )
        incoming_counts = np.zeros((num_intersections, max_lanes), dtype=np.float32)
        incoming_waiting = np.zeros((num_intersections, max_lanes), dtype=np.float32)
        lane_mask = np.zeros((num_intersections, max_lanes), dtype=np.float32)
        action_mask = np.ones((num_intersections, 2), dtype=np.float32)
        current_phase = np.zeros(num_intersections, dtype=np.int64)
        phase_elapsed = np.zeros(num_intersections, dtype=np.float32)
        outgoing_congestion = np.zeros(num_intersections, dtype=np.float32)
        district_type_indices = np.zeros(num_intersections, dtype=np.int64)
        boundary_mask = np.zeros(num_intersections, dtype=np.float32)

        for row_index, intersection_id in enumerate(self.intersection_ids):
            config = self.intersections[intersection_id]
            lane_count_vector, waiting_vector, mask_vector = self._lane_vectors(
                config=config,
                lane_vehicle_count=lane_vehicle_count,
                lane_waiting_count=lane_waiting_count,
            )
            incoming_counts[row_index] = lane_count_vector
            incoming_waiting[row_index] = waiting_vector
            lane_mask[row_index] = mask_vector

            phase_index = int(phase_positions[intersection_id])
            phase_time = float(phase_elapsed_times[intersection_id])
            phase_count = max(1, config.num_green_phases)
            current_phase[row_index] = phase_index
            phase_elapsed[row_index] = phase_time
            district_type_indices[row_index] = config.district_type_index
            boundary_mask[row_index] = 1.0 if config.is_boundary else 0.0

            next_col = 0
            observations[row_index, next_col : next_col + max_lanes] = (
                lane_count_vector / self.config.count_scale
            )
            next_col += max_lanes
            observations[row_index, next_col : next_col + max_lanes] = (
                waiting_vector / self.config.count_scale
            )
            next_col += max_lanes
            observations[row_index, next_col : next_col + max_lanes] = mask_vector
            next_col += max_lanes

            if self.config.include_outgoing_congestion:
                outgoing_congestion[row_index] = self._mean_outgoing_congestion(
                    config=config,
                    lane_vehicle_count=lane_vehicle_count,
                )

            meta_features = [
                normalize_scalar(phase_index, max(1, phase_count - 1))
                if phase_count > 1
                else 0.0,
                normalize_scalar(phase_time, self.config.elapsed_time_scale),
                normalize_scalar(float(outgoing_congestion[row_index]), self.config.count_scale),
                normalize_scalar(float(lane_count_vector.sum()), self.config.count_scale),
                normalize_scalar(float(phase_count), 4.0),
                1.0 if switch_allowed[intersection_id] else 0.0,
                boundary_mask[row_index],
            ]
            observations[row_index, next_col : next_col + len(meta_features)] = meta_features
            next_col += len(meta_features)

            if self.config.include_district_type_feature:
                observations[row_index, next_col + config.district_type_index] = 1.0
                next_col += 4

            if self.config.include_district_context:
                district_values = district_context.get(
                    config.district_id,
                    (0.0, 0.0),
                )
                observations[row_index, next_col : next_col + len(district_values)] = district_values

            if not switch_allowed[intersection_id]:
                action_mask[row_index, 1] = 0.0

        return {
            "observations": observations,
            "incoming_counts": incoming_counts,
            "incoming_waiting": incoming_waiting,
            "lane_mask": lane_mask,
            "action_mask": action_mask,
            "current_phase": current_phase,
            "phase_elapsed": phase_elapsed,
            "outgoing_congestion": outgoing_congestion,
            "boundary_mask": boundary_mask,
            "district_type_indices": district_type_indices,
            "district_types": tuple(
                self.intersections[intersection_id].district_type
                for intersection_id in self.intersection_ids
            ),
            "district_ids": tuple(
                self.intersections[intersection_id].district_id
                for intersection_id in self.intersection_ids
            ),
            "intersection_ids": self.intersection_ids,
        }

    def _compute_observation_dim(self) -> int:
        base_dim = self.config.max_incoming_lanes * 3 + 7
        if self.config.include_district_type_feature:
            base_dim += 4
        if self.config.include_district_context:
            base_dim += 2
        return base_dim

    def _lane_vectors(
        self,
        config: IntersectionConfig,
        lane_vehicle_count: dict[str, int],
        lane_waiting_count: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_lanes = self.config.max_incoming_lanes
        count_vector = np.zeros(max_lanes, dtype=np.float32)
        waiting_vector = np.zeros(max_lanes, dtype=np.float32)
        mask_vector = np.zeros(max_lanes, dtype=np.float32)

        for lane_index, lane_id in enumerate(config.incoming_lanes[:max_lanes]):
            count_vector[lane_index] = float(lane_vehicle_count.get(lane_id, 0))
            waiting_vector[lane_index] = float(lane_waiting_count.get(lane_id, 0))
            mask_vector[lane_index] = 1.0

        return count_vector, waiting_vector, mask_vector

    def _mean_outgoing_congestion(
        self,
        config: IntersectionConfig,
        lane_vehicle_count: dict[str, int],
    ) -> float:
        if not config.outgoing_lanes:
            return 0.0
        total = sum(float(lane_vehicle_count.get(lane_id, 0)) for lane_id in config.outgoing_lanes)
        return total / float(len(config.outgoing_lanes))

    def _compute_district_context(
        self,
        lane_vehicle_count: dict[str, int],
        lane_waiting_count: dict[str, int],
    ) -> dict[str, tuple[float, float]]:
        context: dict[str, tuple[float, float]] = {}
        if not self.config.include_district_context:
            return context

        for district_id, district in self.districts.items():
            total_count = 0.0
            total_waiting = 0.0
            for intersection_id in district.intersection_ids:
                config = self.intersections[intersection_id]
                total_count += sum(
                    float(lane_vehicle_count.get(lane_id, 0))
                    for lane_id in config.incoming_lanes
                )
                total_waiting += sum(
                    float(lane_waiting_count.get(lane_id, 0))
                    for lane_id in config.incoming_lanes
                )

            size = float(self._district_sizes[district_id])
            context[district_id] = (
                normalize_scalar(total_count / size, self.config.count_scale),
                normalize_scalar(total_waiting / size, self.config.count_scale),
            )

        return context
