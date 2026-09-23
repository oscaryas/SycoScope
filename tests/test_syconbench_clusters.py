from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.analyze_probes import cluster_syconbench_stability as cluster
from probing.analyze_probes.analyze_probes import cluster_sweep


class ClusterTests(unittest.TestCase):
    def test_group_bootstrap_preserves_all_turns_and_multiplicity(self):
        groups = np.repeat(np.arange(12), 5)
        draw = cluster.group_draw(groups, np.random.default_rng(1))
        self.assertEqual(len(draw), len(groups))
        for group in np.unique(groups):
            counts = np.bincount(draw, minlength=len(groups))[groups == group]
            self.assertTrue(np.all(counts == counts[0]))

    def test_residuals_remove_modeled_nuisances(self):
        rng = np.random.default_rng(4)
        turns = np.tile(np.arange(1, 6), 20)
        lengths = rng.integers(10, 500, len(turns))
        values = rng.normal(size=(len(turns), 8)) + np.log1p(lengths[:, None]) + turns[:, None]
        residual = cluster.residualize(values, turns, lengths)
        for turn in range(1, 6):
            np.testing.assert_allclose(residual[turns == turn].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.log1p(lengths) @ residual, 0, atol=1e-10)

    def test_matches_existing_sweep(self):
        values = np.random.default_rng(5).normal(size=(200, 20))
        values[:, :6] += values[:, 6, None] * 3
        corr = cluster.correlation(values)
        names = list(map(str, range(20)))
        old = cluster_sweep(names, corr, range(2, 10), signed=True)
        best = max(old, key=lambda k: old[k]["silhouette"])
        labels = [next(j for j, members in enumerate(old[best]["clusters"].values()) if n in members) for n in names]
        new = cluster.partition(corr)
        self.assertEqual(int(best), new["best_k"])
        self.assertEqual(adjusted_rand_score(labels, new["labels"]), 1)

    def test_constant_scores_fail_loudly(self):
        with self.assertRaises(ValueError):
            cluster.correlation(np.ones((20, 10)))


if __name__ == "__main__":
    unittest.main()
