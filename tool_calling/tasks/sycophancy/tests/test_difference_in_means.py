import sys
import unittest
from pathlib import Path

import numpy as np

SYCOPHANCY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SYCOPHANCY_DIR.parents[2]
for p in (REPO_ROOT, SYCOPHANCY_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from difference_in_means.run_dim import compute_layer


class RunDimTests(unittest.TestCase):
    """Ported from building-agent's original
    tool_calling/tasks/sycophancy/pipeline_scripts/tests/test_pipeline.py::
    test_dim_uses_group_cv_and_midpoint_threshold, now exercised against
    compute_layer at its relocated home (difference_in_means/run_dim.py,
    moved out of pipeline_scripts/ during the shared-probing-library
    extraction; this specific grouped-CV DIM driver stays application-owned
    because run_dim.py optionally invokes steering_eval's live-steering
    evaluator -- see probing/probe/dim.py for the pure DIM-direction
    function that WAS extracted into the shared package, covered by
    tests/test_baseline_probe_pipeline.py instead)."""

    def test_dim_uses_group_cv_and_midpoint_threshold(self):
        rng = np.random.default_rng(0)
        labels = np.asarray([0] * 10 + [1] * 10)
        groups = np.asarray([f"g{i}" for i in range(20)], dtype=object)
        features = rng.normal(size=(20, 4)) + labels[:, None] * 2
        result = compute_layer(features, labels, groups, folds=5, seed=0)
        self.assertIn("cv_cohens_d_ci", result)
        self.assertGreater(result["cv_accuracy_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
