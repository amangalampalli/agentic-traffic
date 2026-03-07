class DistrictSummaryBuilder:
    def build(self, adapter, district_config):
        waiting = adapter.get_lane_waiting_vehicle_count()

        return {
            "district_id": district_config.id,
            "intersection_ids": district_config.intersection_ids,
            "avg_wait": sum(waiting.values()) / len(waiting),
        }
