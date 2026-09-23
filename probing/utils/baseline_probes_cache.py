from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from probing.utils.baseline_probes_common import json_dump, read_jsonl, write_jsonl


@dataclass
class ActivationCache:
    path: Path
    metadata: dict
    records: list[dict]
    arrays: dict[str, np.ndarray]

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([record["label"] for record in self.records], dtype=np.int64)

    @property
    def groups(self) -> np.ndarray:
        return np.asarray([str(record["group_id"]) for record in self.records], dtype=object)


def save_activation_cache(
    output_dir: Path,
    metadata: dict,
    records: list[dict],
    arrays: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    for component, array in arrays.items():
        expected_axis = 2 if component == "mha" else 1
        if array.shape[expected_axis] != n:
            raise ValueError(
                f"{component} example axis has {array.shape[expected_axis]} rows; records has {n}"
            )
        np.save(output_dir / f"{component}.npy", array)
    write_jsonl(output_dir / "records.jsonl", records)
    json_dump(output_dir / "metadata.json", metadata)


def load_activation_cache(path: Path, expected_model: str | None = None) -> ActivationCache:
    import json

    metadata_path = path / "metadata.json"
    records_path = path / "records.jsonl"
    if not metadata_path.exists() or not records_path.exists():
        raise FileNotFoundError(f"{path} is not an activation cache")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if expected_model is not None and metadata.get("model") != expected_model:
        raise ValueError(
            f"cache model mismatch: expected {expected_model!r}, found {metadata.get('model')!r} in {path}"
        )
    records = read_jsonl(records_path)
    arrays = {}
    for component in metadata.get("components", []):
        array_path = path / f"{component}.npy"
        if not array_path.exists():
            raise FileNotFoundError(f"metadata lists {component}, but {array_path} is missing")
        arrays[component] = np.load(array_path, mmap_mode="r")
    if "residual" not in arrays and not arrays:
        raise ValueError(f"activation cache {path} has no component arrays")
    return ActivationCache(path, metadata, records, arrays)

