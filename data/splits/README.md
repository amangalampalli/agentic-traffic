# data/splits

City-level train/validation/test splits for the generated dataset.

## Files

- [train_cities.txt](/Users/aditya/Developer/traffic-llm/data/splits/train_cities.txt)
- [val_cities.txt](/Users/aditya/Developer/traffic-llm/data/splits/val_cities.txt)
- [test_cities.txt](/Users/aditya/Developer/traffic-llm/data/splits/test_cities.txt)

## Important rule

Splits are by city only. All scenarios for a given city belong to the same split.

## Regeneration

Use:

`python3 -m training.train_local_policy make-splits`

The split logic is implemented in [training/dataset.py](/Users/aditya/Developer/traffic-llm/training/dataset.py).
