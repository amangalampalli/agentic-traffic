from __future__ import annotations

from pathlib import Path

def load_jsonl_text_dataset(
    path: str | Path,
    controller_families: list[str] | None = None,
    controller_types: list[str] | None = None,
):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(Path(path)), split="train")
    if "text" not in dataset.column_names:
        raise ValueError("Expected a JSONL dataset with a 'text' field.")
    if controller_families:
        allowed_families = set(controller_families)
        dataset = dataset.filter(
            lambda row: row.get("controller_family") in allowed_families
        )
    if controller_types:
        allowed_types = set(controller_types)
        dataset = dataset.filter(
            lambda row: row.get("controller_type") in allowed_types
        )
    if len(dataset) == 0:
        raise ValueError("No dataset rows remain after applying the requested filters.")
    return dataset
