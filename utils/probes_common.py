from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


def parse_dataset_spec(value: str) -> DatasetSpec:
    if "=" not in value:
        raise ValueError(f"dataset must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"dataset must be NAME=PATH, got {value!r}")
    return DatasetSpec(name.strip(), Path(path).expanduser())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_dump(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def t_interval(
    values: list[float], confidence: float = 0.95, bounds: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Student-t interval over fold statistics without requiring scipy."""
    if not values:
        return (0.0, 0.0)
    mean = float(np.mean(values))
    if len(values) == 1:
        return (mean, mean)
    # Two-sided 95% critical values. CV normally uses five folds, but the
    # complete small-df table keeps configurable fold counts honest.
    t95 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if confidence != 0.95:
        raise ValueError("only 95% CV confidence intervals are supported")
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    margin = t95.get(len(values) - 1, 1.96) * sem
    low, high = mean - margin, mean + margin
    if bounds is not None:
        low, high = max(bounds[0], low), min(bounds[1], high)
    return (low, high)


def t_confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Student-t interval for a metric bounded to [0, 1]."""
    return t_interval(values, confidence, bounds=(0.0, 1.0))


def deterministic_indices(indices: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n >= len(indices):
        return indices.copy()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=n, replace=False))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
