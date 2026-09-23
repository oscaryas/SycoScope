from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompt_probes/pipeline"))
from activation_variance import decompose, bootstrap_kernels, resampled_shares, question_bootstrap


class ActivationVarianceTests(unittest.TestCase):
    def test_orthogonal_components_recovered(self):
        a = np.array([-2., 0., 2.])[:, None]
        b = np.array([-3., -1., 1., 3.])[None, :]
        x = np.stack(np.broadcast_arrays(a, b, a*b), axis=2)
        r = decompose(x)
        expected = [4 * float((a*a).sum()), 3 * float((b*b).sum()), float(((a*b)**2).sum())]
        np.testing.assert_allclose(list(r["ss"].values()), expected)
        self.assertAlmostEqual(sum(r["pct"].values()), 100)

    def test_pure_system_and_pure_question(self):
        r = np.random.default_rng(5)
        system = np.repeat(r.normal(size=(4, 1, 9)), 7, axis=1)
        question = np.repeat(r.normal(size=(1, 7, 9)), 4, axis=0)
        self.assertAlmostEqual(decompose(system)["pct"]["system"], 100)
        self.assertAlmostEqual(decompose(question)["pct"]["question"], 100)

    def test_kernel_bootstrap_equals_explicit_resampling(self):
        rng = np.random.default_rng(20)
        x = rng.normal(size=(4, 12, 7)) + rng.normal(size=(4, 1, 7))
        kernels = bootstrap_kernels(x)
        for _ in range(8):
            ix = rng.integers(0, 12, size=12)
            counts = np.bincount(ix, minlength=12)
            actual = resampled_shares(kernels, counts)[0]
            expected = list(decompose(x[:, ix])["pct"].values())
            np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_translation_scale_rotation_invariance(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(4, 11, 7))
        rotation, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        before = list(decompose(x)["pct"].values())
        after = list(decompose((x @ rotation) * 5 + 10)["pct"].values())
        np.testing.assert_allclose(before, after, atol=1e-10)

    def test_reproducible_intervals_and_invalid_data(self):
        x = np.random.default_rng(10).normal(size=(3, 8, 5))
        self.assertEqual(question_bootstrap(x, 20), question_bootstrap(x, 20))
        for invalid in (np.ones((3, 8, 5)), np.full((3, 8, 5), np.nan)):
            with self.assertRaises(ValueError):
                decompose(invalid)


if __name__ == "__main__":
    unittest.main()
