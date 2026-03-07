# env

CityFlow environment implementation for intersection-level RL with district-type metadata.

## Main files

- [traffic_env.py](/Users/aditya/Developer/traffic-llm/env/traffic_env.py)
  Main environment. One episode corresponds to one `(city, scenario)` pair.
- [cityflow_adapter.py](/Users/aditya/Developer/traffic-llm/env/cityflow_adapter.py)
  Thin wrapper around the CityFlow Python engine.
- [observation_builder.py](/Users/aditya/Developer/traffic-llm/env/observation_builder.py)
  Converts variable city topology into fixed-size per-intersection tensors.
- [reward.py](/Users/aditya/Developer/traffic-llm/env/reward.py)
  Configurable local reward calculation.
- [utils.py](/Users/aditya/Developer/traffic-llm/env/utils.py)
  Topology parsing and helper functions.
- [intersection_config.py](/Users/aditya/Developer/traffic-llm/env/intersection_config.py)
  Internal topology dataclasses.

## How it works

- Reads `roadnet.json`, `district_map.json`, and district types from `metadata.json`.
- Identifies non-virtual controllable intersections with at least two green phases.
- Uses one action per controllable intersection.
- Enforces `min_green_time` inside the environment.
- Advances CityFlow for `decision_interval` simulator steps between policy decisions.
- Returns a batched observation for all controlled intersections.

## Observation model

Per intersection:

- padded incoming lane vehicle counts
- padded incoming lane waiting counts
- incoming lane mask
- current green phase index
- elapsed time in current phase
- optional outgoing congestion summary
- district-type one-hot features
- optional small district context
- boundary-intersection indicator

The observation dimension is exposed as `TrafficEnv.observation_dim`.
