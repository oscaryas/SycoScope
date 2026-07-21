import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SAE.pipeline import label_clusters as lc
from utils.llm_judge import parse_json_response


def make_sentences_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_idx": np.arange(n),
            "text": [f"sentence {i}" for i in range(n)],
        }
    )


class TestGatherExemplars(unittest.TestCase):
    def test_basic(self):
        sentences_df = make_sentences_df(10)
        assignments = np.array([0, 1, 0, 1, 0, -1, 1, 0, 1, 0])
        strength = np.array([0.5, 0.1, 0.9, 0.2, 0.3, 0.0, 0.4, 0.7, 0.6, 0.8])
        top, random_ = lc.gather_exemplars(assignments, strength, sentences_df, latent_id=0, n_top=2, n_random=2)
        # members of latent 0: idx 0,2,4,7,9 with strengths 0.5,0.9,0.3,0.7,0.8 -> top2 = idx2,idx9
        self.assertEqual(top, ["sentence 2", "sentence 9"])
        self.assertEqual(len(random_), 2)
        self.assertTrue(set(random_).issubset({"sentence 0", "sentence 4", "sentence 7"}))

    def test_empty_cluster(self):
        sentences_df = make_sentences_df(5)
        assignments = np.array([1, 1, 1, 1, 1])
        strength = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        top, random_ = lc.gather_exemplars(assignments, strength, sentences_df, latent_id=0)
        self.assertEqual(top, [])
        self.assertEqual(random_, [])

    def test_cluster_smaller_than_n_top(self):
        sentences_df = make_sentences_df(6)
        assignments = np.array([0, 0, 0, 1, 1, 1])
        strength = np.array([0.3, 0.1, 0.2, 0.9, 0.9, 0.9])
        top, random_ = lc.gather_exemplars(assignments, strength, sentences_df, latent_id=0, n_top=100, n_random=100)
        self.assertEqual(set(top), {"sentence 0", "sentence 1", "sentence 2"})
        self.assertEqual(random_, [])

    def test_top_and_random_disjoint(self):
        sentences_df = make_sentences_df(50)
        rng = np.random.default_rng(1)
        assignments = rng.integers(0, 3, size=50)
        strength = rng.random(50)
        top, random_ = lc.gather_exemplars(assignments, strength, sentences_df, latent_id=0, n_top=3, n_random=100)
        self.assertEqual(set(top) & set(random_), set())


class TestParseJsonResponse(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_json_response('{"title": "a", "description": "b"}'), {"title": "a", "description": "b"})

    def test_fenced_json(self):
        text = '```json\n{"title": "a", "description": "b"}\n```'
        self.assertEqual(parse_json_response(text), {"title": "a", "description": "b"})

    def test_json_with_surrounding_prose(self):
        text = 'Sure, here is the label:\n{"title": "a", "description": "b"}\nHope that helps!'
        self.assertEqual(parse_json_response(text), {"title": "a", "description": "b"})

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            parse_json_response("no json here at all")


class TestLabelLatent(unittest.TestCase):
    def setUp(self):
        self.sentences_df = make_sentences_df(10)
        self.assignments = np.array([0, 0, 1, 1, 1, -1, -1, -1, -1, -1])
        self.strength = np.array([0.5, 0.6, 0.1, 0.2, 0.3, 0, 0, 0, 0, 0])

    def test_skips_empty_cluster_without_calling_judge(self):
        with patch.object(lc, "call_judge", side_effect=AssertionError("should not be called")):
            record = lc.label_latent(2, np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]),
                                      make_sentences_df(3), n_top=100, n_random=100, seed=0, judge_model="x")
        self.assertEqual(record["status"], "skipped_empty_cluster")
        self.assertEqual(record["cluster_size"], 0)

    def test_success(self):
        with patch.object(lc, "call_judge", return_value='{"title": "Hedging", "description": "desc"}'):
            record = lc.label_latent(0, self.assignments, self.strength, self.sentences_df,
                                      n_top=100, n_random=100, seed=0, judge_model="x")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["title"], "Hedging")
        self.assertEqual(record["cluster_size"], 2)

    def test_judge_failure_is_isolated(self):
        with patch.object(lc, "call_judge", side_effect=RuntimeError("boom")):
            record = lc.label_latent(0, self.assignments, self.strength, self.sentences_df,
                                      n_top=100, n_random=100, seed=0, judge_model="x")
        self.assertEqual(record["status"], "error")
        self.assertIn("boom", record["error"])
        self.assertIsNone(record["title"])

    def test_malformed_judge_json_is_isolated(self):
        with patch.object(lc, "call_judge", return_value='{"title": ""}'):
            record = lc.label_latent(0, self.assignments, self.strength, self.sentences_df,
                                      n_top=100, n_random=100, seed=0, judge_model="x")
        self.assertEqual(record["status"], "error")


class TestLabelRunResumability(unittest.TestCase):
    def _make_run_dir(self, tmp: Path, n_latents=3, n_sentences=9):
        run_dir = tmp / "L99_n3_k3_uncentered_s0"
        run_dir.mkdir()
        config = {"layer": 99, "n_latents": n_latents, "k": 3, "centered": False, "seed": 0}
        (run_dir / "config.json").write_text(json.dumps(config))
        # latent 0: members 0,1,2 ; latent 1: no members (empty) ; latent 2: members 3,4
        assignments = np.array([0, 0, 0, 2, 2, -1, -1, -1, -1], dtype=np.int32)
        strength = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0, 0, 0, 0], dtype=np.float32)
        np.save(run_dir / "assignments.npy", assignments)
        np.save(run_dir / "assignment_strength.npy", strength)
        return run_dir

    def test_rerun_only_retries_failed_latents(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir = self._make_run_dir(tmp)
            sentences_df = make_sentences_df(9)

            call_count = {"n": 0}

            def flaky_judge(prompt, system=None, model="x"):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("transient failure")
                return '{"title": "T", "description": "D"}'

            with patch.object(lc, "call_judge", side_effect=flaky_judge):
                lc.label_run(run_dir, sentences_df, judge_model="x")

            records = json.loads((run_dir / "labels.json").read_text())
            self.assertEqual(len(records), 3)
            statuses = {r["latent_id"]: r["status"] for r in records}
            self.assertEqual(statuses[1], "skipped_empty_cluster")
            # exactly one of latent 0 / latent 2 failed on the first (flaky) call
            self.assertEqual(sorted(statuses.values()), ["error", "ok", "skipped_empty_cluster"])

            first_call_count = call_count["n"]

            def always_ok_judge(prompt, system=None, model="x"):
                call_count["n"] += 1
                return '{"title": "T2", "description": "D2"}'

            with patch.object(lc, "call_judge", side_effect=always_ok_judge):
                lc.label_run(run_dir, sentences_df, judge_model="x")

            self.assertEqual(call_count["n"] - first_call_count, 1)
            records2 = json.loads((run_dir / "labels.json").read_text())
            statuses2 = {r["latent_id"]: r["status"] for r in records2}
            self.assertEqual(sorted(statuses2.values()), ["ok", "ok", "skipped_empty_cluster"])

    def test_force_relabels_everything(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir = self._make_run_dir(tmp)
            sentences_df = make_sentences_df(9)
            call_count = {"n": 0}

            def counting_judge(prompt, system=None, model="x"):
                call_count["n"] += 1
                return '{"title": "T", "description": "D"}'

            with patch.object(lc, "call_judge", side_effect=counting_judge):
                lc.label_run(run_dir, sentences_df, judge_model="x")
            first = call_count["n"]
            with patch.object(lc, "call_judge", side_effect=counting_judge):
                lc.label_run(run_dir, sentences_df, judge_model="x", force=True)
            # force relabels both non-empty latents (0 and 2) again
            self.assertEqual(call_count["n"] - first, 2)


class TestExpandLayerRuns(unittest.TestCase):
    def test_skips_non_run_dirs(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            real = tmp / "L18_n5_k3_uncentered_s0"
            real.mkdir()
            (real / "config.json").write_text("{}")
            stray = tmp / "L18_n5_k3_uncentered_s0" / ".ipynb_checkpoints"
            stray.mkdir()
            other_layer = tmp / "L09_n5_k3_uncentered_s0"
            other_layer.mkdir()
            (other_layer / "config.json").write_text("{}")

            result = lc.expand_layer_runs(18, tmp)
            self.assertEqual(result, [real])


class TestLabelRunAgainstRealRunDir(unittest.TestCase):
    def test_real_run_dir_shape(self):
        real_run_dir = REPO_ROOT / "SAE" / "results" / "trained_sae" / "L18_n5_k3_uncentered_s0"
        if not real_run_dir.exists():
            self.skipTest("real run dir not present in this checkout")

        with tempfile.TemporaryDirectory() as tmp_str:
            copy_dir = Path(tmp_str) / "run"
            shutil.copytree(real_run_dir, copy_dir)
            (copy_dir / "labels.json").unlink(missing_ok=True)

            n_sentences = len(np.load(copy_dir / "assignments.npy"))
            sentences_df = make_sentences_df(n_sentences)

            with patch.object(lc, "call_judge", return_value='{"title": "T", "description": "D"}'):
                lc.label_run(copy_dir, sentences_df, judge_model="x")

            records = json.loads((copy_dir / "labels.json").read_text())
            self.assertEqual(len(records), 5)
            for i, rec in enumerate(records):
                self.assertEqual(rec["latent_id"], i)
                self.assertIn(rec["status"], ("ok", "skipped_empty_cluster"))


if __name__ == "__main__":
    unittest.main()
