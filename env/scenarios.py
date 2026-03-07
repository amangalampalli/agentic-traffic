class ScenarioGenerator:
    def generate(self, seed):
        import random

        random.seed(seed)

        return {
            "traffic_bias": random.choice(["ns", "ew", "balanced"]),
            "emergency_vehicle": random.random() < 0.2,
        }
