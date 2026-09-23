from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SYCOPHANCY_DIR = Path(__file__).resolve().parents[2]
if str(SYCOPHANCY_DIR) not in sys.path:
    sys.path.insert(0, str(SYCOPHANCY_DIR))

from pipeline_scripts.cache import ActivationCache
from pipeline_scripts.common import t_confidence_interval, t_interval
from pipeline_scripts.datasets import prepare_cache
from pipeline_scripts.difference_in_means.run_dim import compute_layer
from pipeline_scripts.generations.generate_syconbench import (
    DEFAULT_SOURCE, load_debate, load_ethical, load_false_presupposition,
)
from pipeline_scripts.judges.scoring import score_rows
from pipeline_scripts.training import Bundle, equalize_and_pool, reserve_group_holdout


class PipelineTests(unittest.TestCase):
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

    def test_dim_uses_group_cv_and_midpoint_threshold(self):
        rng = np.random.default_rng(0)
        labels = np.asarray([0] * 10 + [1] * 10)
        groups = np.asarray([f"g{i}" for i in range(20)], dtype=object)
        features = rng.normal(size=(20, 4)) + labels[:, None] * 2
        result = compute_layer(features, labels, groups, folds=5, seed=0)
        self.assertIn("cv_cohens_d_ci", result)
        self.assertGreater(result["cv_accuracy_mean"], 0.5)

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


if __name__ == "__main__":
    unittest.main()
