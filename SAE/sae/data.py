"""
Reads the frozen activation cache written by SAE/pipeline/cache_activations.py
(sentences.parquet / responses.parquet, layer_XX.npy /
response_means_layer_XX.npy memmaps, meta.json) for SAE training. Splits by
response_id, not by row, so sentences from the same response never straddle
train/val.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "activations"


def load_cache_meta(cache_dir) -> dict:
    import json

    return json.loads((Path(cache_dir) / "meta.json").read_text(encoding="utf-8"))


def _memmap_path(cache_dir: Path, layer: int, granularity: str) -> Path:
    if granularity == "sentence":
        return Path(cache_dir) / f"layer_{layer:02d}.npy"
    if granularity == "response":
        return Path(cache_dir) / f"response_means_layer_{layer:02d}.npy"
    raise ValueError(f"granularity must be 'sentence' or 'response', got {granularity!r}")


class ActivationSplit:
    """Read-only view over one layer's cached activations, split by response_id."""

    def __init__(self, cache_dir, layer: int, granularity: str, val_frac: float = 0.1, seed: int = 0):
        cache_dir = Path(cache_dir)
        self.meta = load_cache_meta(cache_dir)
        if layer not in self.meta["layers"]:
            raise ValueError(f"layer {layer} not in cached layers {self.meta['layers']}")

        self.granularity = granularity
        self.hidden_size = self.meta["hidden_size"]
        self.activations = np.lib.format.open_memmap(_memmap_path(cache_dir, layer, granularity), mode="r")

        if granularity == "sentence":
            table = pd.read_parquet(cache_dir / "sentences.parquet", columns=["response_id", "global_idx"])
            row_response_ids = table.set_index("global_idx").sort_index()["response_id"].to_numpy()
        else:
            table = pd.read_parquet(cache_dir / "responses.parquet", columns=["response_id", "response_idx"])
            row_response_ids = table.set_index("response_idx").sort_index()["response_id"].to_numpy()

        if len(row_response_ids) != self.activations.shape[0]:
            raise ValueError(
                f"row count mismatch: {len(row_response_ids)} rows in parquet vs "
                f"{self.activations.shape[0]} rows in {_memmap_path(cache_dir, layer, granularity).name}"
            )

        unique_responses = np.unique(row_response_ids)
        shuffled = np.random.default_rng(seed).permutation(unique_responses)
        n_val = max(1, int(len(shuffled) * val_frac))
        val_responses = set(shuffled[:n_val])

        is_val = pd.Series(row_response_ids).isin(val_responses).to_numpy()
        self.train_idx = np.nonzero(~is_val)[0]
        self.val_idx = np.nonzero(is_val)[0]

    def iter_batches(self, split: str, batch_size: int, shuffle: bool, seed: int = 0):
        idx = self.train_idx if split == "train" else self.val_idx
        if shuffle:
            idx = np.random.default_rng(seed).permutation(idx)
        for start in range(0, len(idx), batch_size):
            batch_idx = np.sort(idx[start : start + batch_size])
            yield torch.from_numpy(self.activations[batch_idx].astype(np.float32))

    def compute_stats(self, sample_n: int = 4096, seed: int = 0):
        """
        Returns (scale, mean): scale so that (raw_activation * scale) has mean
        squared norm ~= hidden_size, and mean is the scaled sample mean —
        used together as a stable, precomputed baseline for FVU evaluation.
        """
        idx = self.train_idx
        if len(idx) > sample_n:
            idx = np.random.default_rng(seed).choice(idx, size=sample_n, replace=False)
        idx = np.sort(idx)
        sample = self.activations[idx].astype(np.float32)
        mean_sq_norm = float(np.mean(np.sum(sample**2, axis=-1)))
        scale = 1.0 if mean_sq_norm <= 0 else float(np.sqrt(self.hidden_size / mean_sq_norm))
        mean = sample.mean(axis=0) * scale
        return scale, mean
