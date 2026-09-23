import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from probing.evaluations.prompt_probes.judge import judge_behaviors as judge


def annotation():
    return {
        "context": {name: "uncertain" for name in judge.CONTEXT},
        "flags": {"instruction_disclosure": "no", "insufficient_context": "yes"},
        "behaviors": {name: {"presence": "absent", "warrant": "not_applicable",
                              "evidence": [], "reason": "Not shown."}
                      for name in judge.BEHAVIORS},
    }


class BehaviorJudgeTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "secret_pt_explicit", "user_prompt": "I solved it.",
                    "response": "Brilliant!", "system_prompt": "SECRET_INSTRUCTION",
                    "polarity": "sycophantic", "probe_score": 100}

    def test_payload_allowlist(self):
        data = json.loads(judge.payload(self.row))
        self.assertEqual(set(data), {"user_prompt", "response"})
        self.assertNotIn("SECRET_INSTRUCTION", judge.payload(self.row))
        self.assertNotIn("secret_pt_explicit", judge.payload(self.row))

    def test_valid_multilabel_and_uncertainty(self):
        result = annotation()
        for name in ("praise", "enthusiasm"):
            result["behaviors"][name] = {
                "presence": "present", "warrant": "uncertain",
                "evidence": [{"source": "response", "quote": "Brilliant!"}],
                "reason": "No work is visible to assess the praise.",
            }
        self.assertIs(judge.validate(result, self.row), result)

    def test_invalid_quotes_and_inconsistent_ratings(self):
        base = annotation()
        for update in (
            {"presence": "present", "warrant": "uncertain", "evidence": []},
            {"warrant": "unwarranted"},
            {"presence": "present", "warrant": "uncertain",
             "evidence": [{"source": "response", "quote": "Invented praise"}]},
            {"presence": "yes"},
        ):
            with self.subTest(update=update):
                result = copy.deepcopy(base)
                result["behaviors"]["praise"].update(update)
                with self.assertRaises(ValueError):
                    judge.validate(result, self.row)

    def test_style_not_equated_with_unwarrantedness(self):
        result = annotation()
        result["behaviors"]["warmth"].update(
            presence="present", warrant="unwarranted",
            evidence=[{"source": "response", "quote": "Brilliant!"}])
        with self.assertRaises(ValueError):
            judge.validate(result, self.row)

    def test_missing_dimension_rejected(self):
        result = annotation()
        del result["behaviors"]["praise"]
        with self.assertRaises(ValueError):
            judge.validate(result, self.row)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            judge.unique_rows([self.row, self.row])

    def test_prepare_is_blind_stratified_filtered_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            records = [{**self.row, "prompt_id": pid, "polarity": polarity,
                        "example_id": f"{pid}_{polarity}"}
                       for pid in ("train", "test1", "test2")
                       for polarity in ("sycophantic", "non_sycophantic")]
            source.write_text("\n".join(json.dumps(r) for r in records))
            split = root / "split.json"
            split.write_text(json.dumps({"train": ["train"], "test": ["test1", "test2"]}))
            args = argparse.Namespace(inputs=[source], output_dir=root / "out1",
                                      prompt_split=split, split="test", per_stratum=1, seed=0)
            judge.prepare(args)
            first = (args.output_dir / "blind.jsonl").read_text()
            key = json.loads((args.output_dir / "key.json").read_text())
            self.assertEqual(len(key), 2)
            self.assertEqual({r["polarity"] for r in key.values()}, {"sycophantic", "non_sycophantic"})
            self.assertNotIn("train", {r["prompt_id"] for r in key.values()})
            self.assertEqual(set(json.loads(first.splitlines()[0])), {"id", "user_prompt", "response"})
            self.assertNotIn("SECRET_INSTRUCTION", first)
            with self.assertRaises(ValueError):
                judge.prepare(args)
            args.output_dir = root / "out2"
            judge.prepare(args)
            self.assertEqual(first, (args.output_dir / "blind.jsonl").read_text())

    def test_mocked_run_resume_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "blind.jsonl", root / "labels.jsonl"
            source.write_text(json.dumps(self.row) + "\n")
            args = argparse.Namespace(input=source, output=output, model="test-model",
                                      limit=1, max_tokens=8192)
            with patch.object(judge, "call_judge", return_value=json.dumps(annotation())) as api:
                judge.run(args)
                judge.run(args)
                self.assertEqual(api.call_count, 1)
                self.assertNotIn("SECRET_INSTRUCTION", api.call_args.args[0])
            judge.check(argparse.Namespace(input=source, labels=output))
            args.model = "different-model"
            with self.assertRaises(ValueError):
                judge.run(args)
            args.model = "test-model"
            source.write_text(json.dumps({**self.row, "response": "Changed"}) + "\n")
            with self.assertRaises(ValueError):
                judge.run(args)

    def test_invalid_judgment_not_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "blind.jsonl", root / "labels.jsonl"
            source.write_text(json.dumps(self.row) + "\n")
            args = argparse.Namespace(input=source, output=output, model="test-model",
                                      limit=1, max_tokens=8192)
            with patch.object(judge, "call_judge", return_value='{"wrong": true}'):
                with self.assertRaises(ValueError):
                    judge.run(args)
            self.assertEqual(output.read_text(), "")


if __name__ == "__main__":
    unittest.main()
