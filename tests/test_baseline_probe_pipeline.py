"""Ported from building-agent's tool_calling/tasks/sycophancy/pipeline_scripts/tests/
test_pipeline.py, rewritten against the shared probing/ packages per the Task 3 import
mapping table:

    pipeline_scripts.cache                    -> probing.utils.baseline_probes_cache
    pipeline_scripts.common                   -> probing.utils.baseline_probes_common
    pipeline_scripts.datasets                 -> probing.utils.baseline_probes_datasets
    pipeline_scripts.training                 -> probing.probe.baseline_training
    pipeline_scripts.generations.generate_syconbench
                                               -> probing.evaluations.baseline_probes.generation.generate_syconbench
    pipeline_scripts.judges.scoring            -> probing.evaluations.baseline_probes.judge.scoring

One piece of the original file does NOT have a home here: its DIM test imported
`pipeline_scripts.difference_in_means.run_dim.compute_layer`, but `run_dim.py` is one of
the 14 steering-integrated files that stays on `building-agent` and is explicitly NOT
extracted by this migration (see task-3-report.md). Per the Task 6 brief, that test is not
ported by making this shared test import `tool_calling/` -- instead
`test_dim_computation_reports_effect_size_and_cv_diagnostics` below exercises
`probing.probe.dim.compute_dim_direction`, the pure DIM-direction function that *did* get
extracted into the shared package (see probing/probe/dim.py), covering the same underlying
claim (grouped/cross-validated DIM computation on separable synthetic activations yields a
plausible direction and diagnostic metrics) with the function that actually lives here.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils.baseline_probes_cache import ActivationCache
from probing.utils.baseline_probes_common import t_confidence_interval, t_interval
from probing.utils.baseline_probes_datasets import prepare_cache
from probing.probe.dim import compute_dim_direction
from probing.evaluations.baseline_probes.generation.generate_syconbench import (
    DEFAULT_SOURCE, load_debate, load_ethical, load_false_presupposition,
)
from probing.evaluations.baseline_probes.judge.scoring import score_rows
from probing.probe.baseline_training import Bundle, equalize_and_pool, reserve_group_holdout


class PipelineTests(unittest.TestCase):
    """Coverage ported verbatim (module paths aside) from
    tool_calling/tasks/sycophancy/pipeline_scripts/tests/test_pipeline.py."""

    def test_confidence_intervals_clip_only_bounded_metrics(self):
        self.assertEqual(t_confidence_interval([0.9, 1.0])[1], 1.0)
        self.assertGreater(t_interval([2.0, 3.0])[1], 1.0)

    def test_moral_pair_average_priority_and_no_fixed_cap(self):
        records = []
        outcomes = ["both_nta"] * 3 + ["original_nta_flipped_yta"] * 2 + ["both_yta"] * 2
        values = []
        for group, outcome in enumerate(outcomes):
            label = int(outcome == "both_nta")
            for side in range(2):
                records.append({"group_id": str(group), "pair_outcome": outcome, "label": label})
                values.append([group * 10 + side])
        cache = ActivationCache(
            Path("unused"), {"dataset_type": "moral"}, records,
            {"residual": np.asarray(values, dtype=float).reshape(1, len(values), 1)},
        )
        arrays, labels, _, selected = prepare_cache(cache, "moral", seed=4)
        self.assertEqual(len(labels), 6)
        self.assertEqual(int(labels.sum()), 3)
        negative_outcomes = [row["pair_outcome"] for row in selected if row["label"] == 0]
        self.assertEqual(negative_outcomes.count("original_nta_flipped_yta"), 2)
        self.assertEqual(negative_outcomes.count("both_yta"), 1)
        for row_index, record in enumerate(selected):
            group = int(record["group_id"])
            self.assertAlmostEqual(float(arrays["residual"][0, row_index, 0]), group * 10 + 0.5)

    def test_equalized_pool_and_group_holdout_do_not_leak(self):
        def bundle(name, n):
            labels = np.asarray(([0, 1] * n), dtype=int)
            groups = np.asarray([f"{name}:{i}" for i in range(2 * n)], dtype=object)
            array = np.arange(2 * n, dtype=float).reshape(1, 2 * n, 1)
            return Bundle(name, {"residual": array}, labels, groups, [{} for _ in labels])
        pooled = equalize_and_pool([bundle("a", 8), bundle("b", 5)], seed=0)
        self.assertEqual(len(pooled.labels), 20)
        train, holdout = reserve_group_holdout(pooled, 0.2, seed=1)
        self.assertFalse(set(train.groups) & set(holdout.groups))

    def test_dim_computation_reports_effect_size_and_cv_diagnostics(self):
        """Covers the same underlying claim as the original
        test_dim_uses_group_cv_and_midpoint_threshold (a grouped/cross-validated DIM
        computation over separable synthetic activations produces a sane direction plus
        diagnostic metrics), using probing.probe.dim.compute_dim_direction -- the pure
        DIM function that was actually extracted into the shared package. See module
        docstring for why the original's compute_layer (pipeline_scripts.difference_in
        _means.run_dim) is not ported here."""
        rng = np.random.default_rng(0)
        labels = np.asarray([0] * 10 + [1] * 10)
        features = rng.normal(size=(20, 4)) + labels[:, None] * 2
        result = compute_dim_direction(features, labels, method="cv_averaged", n_folds=5, seed=0)
        self.assertIn("fold_effect_sizes", result)
        self.assertIn("auc_roc", result)
        self.assertGreater(result["auc_roc"], 0.5)
        self.assertGreater(result["effect_size"], 0.0)
        self.assertEqual(result["direction"].shape, (4,))

    @unittest.skipUnless(
        (DEFAULT_SOURCE / ".git").exists(),
        "third_party/SYCON-Bench submodule is not checked out in this environment",
    )
    def test_pinned_sycon_sources_are_five_turn(self):
        self.assertTrue((DEFAULT_SOURCE / ".git").exists())
        for loader in (load_debate, load_ethical, load_false_presupposition):
            rows = loader(DEFAULT_SOURCE)
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(len(row["user_turns"]) == 5 for row in rows))

    def test_mcq_judge_excludes_wrong_first_turn_and_labels_caving(self):
        rows = [
            {"answers": "(A) x\n(B) y", "correct_letter": "A", "turn1_response": "Answer: A", "turn2_response": "Answer: B"},
            {"answers": "(A) x\n(B) y", "correct_letter": "A", "turn1_response": "Answer: B", "turn2_response": "Answer: A"},
        ]
        summary, judged = score_rows(rows, "are_you_sure_mcq", "unused")
        self.assertEqual(summary["n_judged"], 1)
        self.assertEqual(summary["rate"], 1.0)
        self.assertIsNone(judged[1]["label"])


class RecursiveImportTests(unittest.TestCase):
    """Recursively imports every module under the moved shared packages, without any
    model/network access, to catch import-time errors (syntax, missing dependency,
    circular import, stray reference to a package that didn't move) across the whole
    surface this migration touched -- not just the handful of modules PipelineTests
    happens to exercise directly."""

    PACKAGES = (
        "probing.utils",
        "probing.probe",
        "probing.evaluations.baseline_probes",
        "probing.analyze_probes",
        "probing.data",
    )

    def test_recursively_imports_moved_packages_without_model_or_network_access(self):
        failures = {}
        for package_name in self.PACKAGES:
            package = importlib.import_module(package_name)
            prefix = package.__name__ + "."
            for _, module_name, _ in pkgutil.walk_packages(package.__path__, prefix):
                try:
                    importlib.import_module(module_name)
                except Exception as exc:  # noqa: BLE001 - we want to report every failure, not stop at the first
                    failures[module_name] = f"{type(exc).__name__}: {exc}"
        if failures:
            details = "\n".join(f"  {name}: {error}" for name, error in sorted(failures.items()))
            self.fail(f"{len(failures)} module(s) under {self.PACKAGES} failed to import:\n{details}")


if __name__ == "__main__":
    unittest.main()
