# openenv_app

OpenEnv-compatible district environment layer.

## Main files

- [app.py](/Users/aditya/Developer/traffic-llm/openenv_app/app.py)
- [openenv_wrapper.py](/Users/aditya/Developer/traffic-llm/openenv_app/openenv_wrapper.py)
- [schema.py](/Users/aditya/Developer/traffic-llm/openenv_app/schema.py)

## Status

This wrapper now sits on top of the active DQN local controller stack. External OpenEnv actions operate at the district level, and the wrapper emits district summaries plus executes slower district decisions over the current CityFlow environment.
