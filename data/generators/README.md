# data/generators

Synthetic dataset generation code for CityFlow cities and scenarios.

## Main files

- [generate_dataset.py](/Users/aditya/Developer/traffic-llm/data/generators/generate_dataset.py)
  Primary dataset build script.
- [city_generator.py](/Users/aditya/Developer/traffic-llm/data/generators/city_generator.py)
  High-level city assembly.
- [roadnet_generator.py](/Users/aditya/Developer/traffic-llm/data/generators/roadnet_generator.py)
  Road network generation.
- [district_generator.py](/Users/aditya/Developer/traffic-llm/data/generators/district_generator.py)
  District assignments and relationships.
- [flow_generator.py](/Users/aditya/Developer/traffic-llm/data/generators/flow_generator.py)
  Vehicle flow generation.
- [scenario_generator.py](/Users/aditya/Developer/traffic-llm/data/generators/scenario_generator.py)
  Scenario-specific flow and config generation.

## Notes

- This folder is for offline dataset creation.
- The current training pipeline consumes the generated files directly and does not regenerate them.
