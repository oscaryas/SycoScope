import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probing.evaluations.prompt_probes.judge import judge_syconbench_budgeted as judge


class BudgetTests(unittest.TestCase):
    def test_cost(self):
        self.assertAlmostEqual(judge.usage_cost({"input_tokens": 1000, "output_tokens": 10}), .0021)

    def run_case(self, budget, fail=False, resume=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jsonl"
            row = {"id": "case", "setting": "debate", "question": "Question?", "stance": "Pro",
                   "messages": [{"role": "assistant", "content": "Response"}] * 5,
                   "turn_finish_reasons": ["stop"] * 5}
            source.write_text(json.dumps(row) + "\n")
            calls = []

            def api(path, body, key):
                if path.endswith("count_tokens"):
                    return {"input_tokens": 1000}
                calls.append(body)
                if fail:
                    raise TimeoutError()
                return {"content": [{"type": "text", "text": "1"}], "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1000, "output_tokens": 1},
                        "model": judge.MODEL, "id": "mock-id"}

            argv = ["judge", "--input", str(source), "--output-dir", str(root / "out"),
                    "--budget-usd", str(budget), "--workers", "3"]
            with patch.object(sys, "argv", argv), patch.object(judge, "load_key", return_value="test"), \
                 patch.object(judge, "api", side_effect=api):
                judge.main()
                if resume:
                    judge.main()
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertLessEqual(summary["accounted_usd"], budget)
            return calls, summary

    def test_cap_prevents_billable_requests(self):
        calls, summary = self.run_case(.001)
        self.assertEqual(len(calls), 0)
        self.assertEqual(summary["n_fully_judged"], 0)

    def test_resume_does_not_rebill(self):
        calls, summary = self.run_case(1, resume=True)
        self.assertEqual(len(calls), 5)
        self.assertEqual(summary["n_fully_judged"], 1)

    def test_unknown_outcome_keeps_reservation(self):
        calls, summary = self.run_case(1, fail=True, resume=True)
        self.assertEqual(len(calls), 5)
        self.assertEqual(summary["n_unknown_outcome"], 5)
        self.assertGreater(summary["accounted_usd"], 0)
        self.assertEqual(summary["known_cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
