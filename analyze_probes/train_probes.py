#!/usr/bin/env python3
"""Train L2-regularized logistic-regression probes with a C sweep from a cached
activation .npz (cache_activations.py or a cache_train_*.py --cache-out). CPU
only. One sweep per cached layer (or the --layers subset): GroupKFold CV
accuracy per C, then scaler + model refit on all rows per C and pickled.

Rows flagged degenerate (empty/repetitive) are dropped together with every
other row in their group, as the prompt-probes pipeline did, unless
--keep-degenerate is set.

Usage:
    python -m analyze_probes.train_probes --cache cache.npz --layers 16 24 \\
        --C-values 0.01 0.1 1.0 10.0 100.0 --output weights.pkl
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from analyze_probes.probes_core import BALANCE_METHODS, DROP_REASONS, load_cache, train_and_save  # noqa: E402

DEFAULT_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, type=Path, help=".npz from probes_core.save_cache")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Default: every layer in the cache.")
    parser.add_argument("--C-values", type=float, nargs="+", default=DEFAULT_C_VALUES)
    parser.add_argument("--balance-method", choices=BALANCE_METHODS, default="undersample")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-degenerate", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cache = load_cache(args.cache)
    layers = args.layers or cache["layers"]
    missing = [layer for layer in layers if layer not in cache["acts"]]
    if missing:
        raise SystemExit(f"layers {missing} not in cache (has {cache['layers']})")

    y, groups = cache["y"], cache["groups"]
    keep = np.ones(len(y), dtype=bool)
    if not args.keep_degenerate and "degenerate" in cache:
        bad = {g for g, d in zip(groups, cache["degenerate"]) if d in DROP_REASONS}
        keep = np.array([g not in bad for g in groups], dtype=bool)
    print(f"{int(keep.sum())}/{len(y)} rows ({int((~keep).sum())} dropped as degenerate)")

    train_and_save(
        {layer: cache["acts"][layer][keep] for layer in layers}, y[keep], groups[keep], layers,
        args.C_values, args.output, seed=args.seed, balance_method=args.balance_method,
        n_splits=args.n_splits, max_iter=args.max_iter,
        meta={"cache": str(args.cache), "n_dropped_degenerate": int((~keep).sum())},
    )


if __name__ == "__main__":
    main()
