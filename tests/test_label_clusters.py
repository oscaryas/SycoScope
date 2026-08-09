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


def make_topk(rows: list[list[int]], strengths: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Build the (n_sentences, k) assignment/strength pair the labeller now consumes."""
    return np.array(rows, dtype=np.int32), np.array(strengths, dtype=np.float32)


class TestGatherExemplars(unittest.TestCase):
    def test_basic(self):
        sentences_df = make_sentences_df(10)
        # latent 0 is in slot 0 of rows 0,2,4,7,9 with strengths 0.5,0.9,0.3,0.7,0.8
        assignments_topk, strength = make_topk(
            [[0, 1], [1, -1], [0, 1], [1, -1], [0, 1], [-1, -1], [1, -1], [0, 1], [1, -1], [0, 1]],
            [[0.5, 0.05], [0.1, 0.0], [0.9, 0.05], [0.2, 0.0], [0.3, 0.05],
             [0.0, 0.0], [0.4, 0.0], [0.7, 0.05], [0.6, 0.0], [0.8, 0.05]],
        )
        top, random_ = lc.gather_exemplars(assignments_topk, strength, sentences_df, latent_id=0, n_top=2, n_random=2)
        self.assertEqual(top, ["sentence 2", "sentence 9"])
        self.assertEqual(len(random_), 2)
        self.assertTrue(set(random_).issubset({"sentence 0", "sentence 4", "sentence 7"}))

    def test_ranks_by_this_latents_own_strength_not_the_slot_winner(self):
        """The point of the top-k switch: a latent is ranked on its own activation even
        where another latent is stronger on the same sentence."""
        sentences_df = make_sentences_df(3)
        # latent 7 sits in slot 1 everywhere, always beaten by the slot-0 latent.
        assignments_topk, strength = make_topk(
            [[0, 7], [0, 7], [0, 7]],
            [[9.0, 0.2], [9.0, 0.8], [9.0, 0.5]],
        )
        top, _ = lc.gather_exemplars(assignments_topk, strength, sentences_df, latent_id=7, n_top=3, n_random=0)
        # ordered by latent 7's own strengths (0.8, 0.5, 0.2), not by the slot-0 values
        self.assertEqual(top, ["sentence 1", "sentence 2", "sentence 0"])

    def test_empty_cluster(self):
        sentences_df = make_sentences_df(5)
        assignments_topk, strength = make_topk(
            [[1, -1]] * 5,
            [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]],
        )
        top, random_ = lc.gather_exemplars(assignments_topk, strength, sentences_df, latent_id=0)
        self.assertEqual(top, [])
        self.assertEqual(random_, [])

    def test_cluster_smaller_than_n_top(self):
        sentences_df = make_sentences_df(6)
        assignments_topk, strength = make_topk(
            [[0, -1], [0, -1], [0, -1], [1, -1], [1, -1], [1, -1]],
            [[0.3, 0.0], [0.1, 0.0], [0.2, 0.0], [0.9, 0.0], [0.9, 0.0], [0.9, 0.0]],
        )
        top, random_ = lc.gather_exemplars(
            assignments_topk, strength, sentences_df, latent_id=0, n_top=100, n_random=100
        )
        self.assertEqual(set(top), {"sentence 0", "sentence 1", "sentence 2"})
        self.assertEqual(random_, [])

    def test_top_and_random_disjoint(self):
        sentences_df = make_sentences_df(50)
        rng = np.random.default_rng(1)
        assignments_topk = rng.integers(0, 3, size=(50, 3)).astype(np.int32)
        strength = rng.random((50, 3)).astype(np.float32)
        top, random_ = lc.gather_exemplars(
            assignments_topk, strength, sentences_df, latent_id=0, n_top=3, n_random=100
        )
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
        # latent 0 in rows 0,1 ; latent 1 in rows 2,3,4 ; rows 5-9 inactive
        self.assignments_topk, self.strength = make_topk(
            [[0, -1], [0, -1], [1, -1], [1, -1], [1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            [[0.5, 0.0], [0.6, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0],
             [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )
        self.argmax = np.array([0, 0, 1, 1, 1, -1, -1, -1, -1, -1], dtype=np.int32)

    def test_skips_empty_cluster_without_calling_judge(self):
        topk, strength = make_topk([[0, -1], [0, -1], [0, -1]], [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        argmax = np.array([0, 0, 0], dtype=np.int32)
        with patch.object(lc, "call_judge", side_effect=AssertionError("should not be called")):
            record = lc.label_latent(2, topk, strength, argmax, make_sentences_df(3),
                                     n_top=100, n_random=100, seed=0, judge_model="x")
        self.assertEqual(record["status"], "skipped_empty_cluster")
        self.assertEqual(record["cluster_size"], 0)
        self.assertEqual(record["n_members_topk"], 0)

    def test_success(self):
        with patch.object(lc, "call_judge", return_value='{"title": "Hedging", "description": "desc"}'):
            record = lc.label_latent(0, self.assignments_topk, self.strength, self.argmax, self.sentences_df,
                                     n_top=100, n_random=100, seed=0, judge_model="x")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["title"], "Hedging")
        self.assertEqual(record["cluster_size"], 2)
        self.assertEqual(record["n_members_topk"], 2)
        self.assertEqual(record["membership"], "topk")

    def test_latent_that_never_wins_argmax_is_still_labelled(self):
        """Previously status would be skipped_empty_cluster: latent 7 has no argmax rows."""
        topk, strength = make_topk(
            [[0, 7], [0, 7], [1, 7]],
            [[9.0, 0.4], [9.0, 0.6], [9.0, 0.5]],
        )
        argmax = np.array([0, 0, 1], dtype=np.int32)
        with patch.object(lc, "call_judge", return_value='{"title": "T", "description": "D"}'):
            record = lc.label_latent(7, topk, strength, argmax, make_sentences_df(3),
                                     n_top=100, n_random=100, seed=0, judge_model="x")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["cluster_size"], 0)
        self.assertEqual(record["n_members_topk"], 3)
        self.assertEqual(record["n_exemplars_used"], 3)

    def test_judge_failure_is_isolated(self):
        with patch.object(lc, "call_judge", side_effect=RuntimeError("boom")):
            record = lc.label_latent(0, self.assignments_topk, self.strength, self.argmax, self.sentences_df,
                                     n_top=100, n_random=100, seed=0, judge_model="x")
        self.assertEqual(record["status"], "error")
        self.assertIn("boom", record["error"])
        self.assertIsNone(record["title"])

    def test_malformed_judge_json_is_isolated(self):
        with patch.object(lc, "call_judge", return_value='{"title": ""}'):
            record = lc.label_latent(0, self.assignments_topk, self.strength, self.argmax, self.sentences_df,
                                     n_top=100, n_random=100, seed=0, judge_model="x")
        self.assertEqual(record["status"], "error")


class TestLabelRunResumability(unittest.TestCase):
    def _make_run_dir(self, tmp: Path, n_latents=3, n_sentences=9):
        run_dir = tmp / "L99_n3_k3_uncentered_s0"
        run_dir.mkdir()
        config = {"layer": 99, "n_latents": n_latents, "k": 2, "centered": False, "seed": 0}
        (run_dir / "config.json").write_text(json.dumps(config))
        # latent 0: members 0,1,2 ; latent 1: no members (empty) ; latent 2: members 3,4
        assignments = np.array([0, 0, 0, 2, 2, -1, -1, -1, -1], dtype=np.int32)
        strength = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0, 0, 0, 0], dtype=np.float32)
        assignments_topk, topk_strength = make_topk(
            [[0, -1], [0, -1], [0, -1], [2, -1], [2, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0],
             [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )
        np.save(run_dir / "assignments.npy", assignments)
        np.save(run_dir / "assignment_strength.npy", strength)
        np.save(run_dir / "assignments_topk.npy", assignments_topk)
        np.save(run_dir / "assignments_topk_strength.npy", topk_strength)
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

    def test_argmax_era_labels_are_relabelled_without_force(self):
        """Records predating the top-k switch have no `membership` key and describe a
        different cluster, so they must not be treated as already done."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir = self._make_run_dir(tmp)
            sentences_df = make_sentences_df(9)

            stale = [
                {"latent_id": i, "cluster_size": 3, "n_exemplars_used": 3, "judge_model": "x",
                 "title": "old", "description": "old", "status": "ok", "error": None}
                for i in range(3)
            ]
            (run_dir / "labels.json").write_text(json.dumps(stale))

            call_count = {"n": 0}

            def counting_judge(prompt, system=None, model="x"):
                call_count["n"] += 1
                return '{"title": "NEW", "description": "NEW"}'

            with patch.object(lc, "call_judge", side_effect=counting_judge):
                lc.label_run(run_dir, sentences_df, judge_model="x")

            self.assertEqual(call_count["n"], 2)  # latents 0 and 2; latent 1 is empty
            records = json.loads((run_dir / "labels.json").read_text())
            self.assertTrue(all(r["membership"] == "topk" for r in records))
            self.assertEqual({r["title"] for r in records if r["status"] == "ok"}, {"NEW"})

    def test_missing_strength_file_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir = self._make_run_dir(tmp)
            (run_dir / "assignments_topk_strength.npy").unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                lc.label_run(run_dir, make_sentences_df(9), judge_model="x")
            self.assertIn("backfill_topk_strength.py", str(ctx.exception))


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
        # L14 rather than L18: the L18 runs' assignments predate a rebuild of the sentence
        # table, so they have no assignments_topk_strength.npy (see backfill_topk_strength.py).
        real_run_dir = REPO_ROOT / "SAE" / "results" / "trained_sae" / "L14_n5_k3_uncentered_s0"
        if not (real_run_dir / "assignments_topk_strength.npy").exists():
            self.skipTest("real run dir not backfilled in this checkout")

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
                # top-k membership can only be a superset of the argmax cluster
                self.assertGreaterEqual(rec["n_members_topk"], rec["cluster_size"])


if __name__ == "__main__":
    unittest.main()
