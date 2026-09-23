import sys
import unittest
from pathlib import Path

import numpy as np

PIPELINE = Path(__file__).resolve().parents[1] / "prompt_probes" / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import train_dim  # noqa: E402


class TestPromptProbeDim(unittest.TestCase):
    def test_holdout_direction_uses_training_prompts_only(self):
        # The held-out pair has an enormous orthogonal shift. If it leaked into
        # fitting, the learned direction would point mostly along dimension 1.
        X = np.array(
            [
                [2.0, 0.0], [0.0, 0.0],
                [3.0, 0.0], [1.0, 0.0],
                [2.0, 100.0], [0.0, -100.0],
            ]
        )
        y = np.array([1, 0, 1, 0, 1, 0])
        prompt_ids = ["a", "a", "b", "b", "c", "c"]
        result = train_dim.fit_holdout_dim(
            X, y, prompt_ids, {"train": ["a", "b"], "test": ["c"]}
        )
        np.testing.assert_allclose(result["direction_raw"], [1.0, 0.0])
        self.assertEqual(result["test_auc"], 1.0)
        self.assertEqual(result["test_paired_win_rate"], 1.0)

    def test_direction_sign_points_to_label_one(self):
        X = np.array([[4.0, 1.0], [0.0, 1.0], [3.0, 1.0], [1.0, 1.0]])
        y = np.array([1, 0, 1, 0])
        result = train_dim.fit_holdout_dim(
            X, y, ["a", "a", "b", "b"], {"train": ["a"], "test": ["b"]}
        )
        self.assertGreater(result["positive_mean"], result["negative_mean"])
        self.assertEqual(result["test_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
