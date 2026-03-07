# agents

Local traffic-control policies and compatibility shims.

## Main files

- [local_policy.py](/Users/aditya/Developer/traffic-llm/agents/local_policy.py)
  Active v1 policy interfaces and simple baselines:
  - `HoldPhasePolicy`
  - `FixedCyclePolicy`
  - `QueueGreedyPolicy`
- [district_controller.py](/Users/aditya/Developer/traffic-llm/agents/district_controller.py)
  Older district-level prototype logic kept for compatibility.
- [district_coordinator.py](/Users/aditya/Developer/traffic-llm/agents/district_coordinator.py)
  Import shim for older code paths.

## Notes

- The learned local-policy network itself lives in [training/models.py](/Users/aditya/Developer/traffic-llm/training/models.py), not here.
- For active training, use the parameter-shared DQN path in `training/`, not the district-controller prototypes.
