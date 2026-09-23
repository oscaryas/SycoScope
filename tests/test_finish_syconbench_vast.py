import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompt_probes/pipeline"))
import finish_syconbench_vast as finish


class SupervisorStatusTests(unittest.TestCase):
    def status(self, code, output):
        result = subprocess.CompletedProcess(["ssh"], code, stdout=output)
        with patch.object(finish.subprocess, "run", return_value=result):
            return finish.supervisor_status(["ssh"], "job")

    def test_stopped_is_valid_with_exit_three(self):
        self.assertEqual(self.status(3, "job STOPPED Not started")[1], "STOPPED")

    def test_running_and_exited(self):
        self.assertEqual(self.status(0, "job RUNNING pid 12")[1], "RUNNING")
        self.assertEqual(self.status(3, "job EXITED yesterday")[1], "EXITED")

    def test_transport_failure_is_not_a_job_state(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.status(255, "")

    def test_unknown_job_is_not_stopped(self):
        with self.assertRaises(RuntimeError):
            self.status(3, "job ERROR (no such process)")


class SavedJudgmentTests(unittest.TestCase):
    def test_completed_and_truncated_inputs_and_mismatch(self):
        rows = [{"id": "a", "messages": [], "model": "model", "question": "q",
                 "setting": "debate", "turn_finish_reasons": ["stop"] * 5}]
        saved = dict(rows[0], turn_judgments=[1] * 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judged.jsonl"
            path.write_text(json.dumps(saved) + "\n")
            finish.validate_saved_judgments(rows, path)
            saved["turn_judgments"][2] = None
            path.write_text(json.dumps(saved) + "\n")
            with self.assertRaises(ValueError):
                finish.validate_saved_judgments(rows, path)
            rows[0]["turn_finish_reasons"][2] = "length"
            path.write_text(json.dumps(saved) + "\n")
            finish.validate_saved_judgments(rows, path)
            saved["question"] = "different"
            path.write_text(json.dumps(saved) + "\n")
            with self.assertRaises(ValueError):
                finish.validate_saved_judgments(rows, path)


if __name__ == "__main__":
    unittest.main()
