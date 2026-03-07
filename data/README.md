# data

Dataset assets for the CityFlow traffic-control project.

## Subfolders

- [generated](/Users/aditya/Developer/traffic-llm/data/generated)
  Pre-generated synthetic cities used for training and evaluation.
- [splits](/Users/aditya/Developer/traffic-llm/data/splits)
  City-level train/val/test split files.
- [generators](/Users/aditya/Developer/traffic-llm/data/generators)
  Dataset generation utilities.

## Dataset contract

Each city directory under `generated/` contains:

- `roadnet.json`
- `district_map.json`
- `metadata.json`
- `scenarios/<scenario_name>/flow.json`
- `scenarios/<scenario_name>/config.json`

The current RL pipeline splits by city, not by scenario.
