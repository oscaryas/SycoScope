import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from probing.utils import common
from probing.data import prompt_wording_control as pwc


class PromptWordingControlTests(unittest.TestCase):
    def test_paraphrases_preserve_cell_identity_and_change_wording(self):
        original = common.load_prompt_pairs()
        paraphrase = common.load_prompt_pairs(pwc.PARAPHRASE_PATH)
        rows = pwc.validate_pairs(original, paraphrase)
        self.assertEqual(len(rows), 20)
        self.assertEqual([row["slug"] for row in rows], [row["slug"] for row in original])
        for row in rows:
            for polarity in common.POLARITIES:
                self.assertLess(row[polarity + "_jaccard"], 0.65)

    def test_prepare_reuses_only_frozen_dolly_split(self):
        pairs = common.load_prompt_pairs()
        prompts = []
        for split, n in (("train", 120), ("report", 80)):
            prompts.extend(
                {
                    "prompt_id": f"dolly-{split}-{i}",
                    "dataset": "dolly",
                    "source": "test",
                    "user_prompt": f"Dolly question {i}",
                    "split": split,
                }
                for i in range(n)
            )
        prompts.append(
            {
                "prompt_id": "perez-0",
                "dataset": "perez",
                "source": "test",
                "user_prompt": "Perez question",
                "split": "report",
            }
        )
        base = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "model_revision": "revision",
            "seed": 7,
            "prompts": prompts,
            "pairs": pairs,
            "layers": list(range(0, 32, 4)),
            "generation": {"temperature": 0.6},
            "analysis": {"C": 1.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_run, run = root / "base", root / "new"
            base_run.mkdir()
            (base_run / "experiment.json").write_text(json.dumps(base), encoding="utf-8")
            pwc.prepare(run, base_run)
            prepared = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual({row["dataset"] for row in prepared["prompts"]}, {"dolly"})
            self.assertEqual(len(prepared["prompts"]), 200)
            self.assertEqual(prepared["n_generation_target"], 8000)
            self.assertEqual(prepared["base_experiment_hash"], pwc.dc.digest(base))

    def test_matrix_medians_keep_diagonal_separate(self):
        matrix = [[0.9, 0.4], [0.6, 0.7]]
        self.assertAlmostEqual(pwc._median(matrix, diagonal=True), 0.8)
        self.assertAlmostEqual(pwc._median(matrix, diagonal=False), 0.5)

    def test_lexical_jaccard(self):
        self.assertEqual(pwc.lexical_jaccard("a b", "a c"), 1 / 3)
        self.assertTrue(np.isfinite(pwc.lexical_jaccard("", "")))


if __name__ == "__main__":
    unittest.main()
