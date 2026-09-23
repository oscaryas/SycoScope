import inspect
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probing.data.sycophancy_data as sycophancy_data
from probing.data.sycophancy_data import (
    DISAGREEMENT_PROMPTS,
    QUESTION_SUFFIX,
    compute_sycophancy_rate,
    detect_capitulation,
    format_chat,
    is_correct,
)


class SycophancyDataImportPathTests(unittest.TestCase):
    """Focused import/path test for probing/data/sycophancy_data.py (moved
    from tool_calling/tasks/sycophancy/sycophancy_data.py on building-agent).

    The only path-shaped dependency this module has is its runtime
    `from gpu_memory import ...` sibling-file import, which had to become
    `from probing.utils.gpu_memory import ...` once gpu_memory.py moved to
    probing/utils/gpu_memory.py -- this is not the same package, so a bare
    `import gpu_memory` would now fail. Assert the rewritten import
    actually appears in main()'s source (main() is not called here since it
    needs a real model + tokenizer).
    """

    def test_main_imports_gpu_memory_from_canonical_location(self):
        source = inspect.getsource(sycophancy_data.main)
        self.assertIn("from probing.utils.gpu_memory import", source)
        self.assertNotIn("from gpu_memory import", source)

    def test_gpu_memory_resolves_from_new_location(self):
        # Import succeeds under the canonical path used by sycophancy_data.main.
        from probing.utils.gpu_memory import clear_gpu_memory, load_model_conservatively

        self.assertTrue(callable(clear_gpu_memory))
        self.assertTrue(callable(load_model_conservatively))


class SycophancyDataPureHelperTests(unittest.TestCase):
    def test_format_chat_falls_back_without_chat_template(self):
        class NoTemplateTokenizer:
            pass

        rendered = format_chat(NoTemplateTokenizer(), [{"role": "user", "content": "hi"}])
        self.assertIn("USER: hi", rendered)
        self.assertTrue(rendered.endswith("\nASSISTANT:"))

    def test_is_correct_matches_keyword_overlap(self):
        self.assertTrue(is_correct("The answer is Paris, the capital of France.", ["Paris"]))
        self.assertFalse(is_correct("I have no idea.", ["Paris"]))

    def test_detect_capitulation_marker_phrase(self):
        label = detect_capitulation("Paris", "London", "You're right, it's London.")
        self.assertEqual(label, 1)

    def test_detect_capitulation_holds_ground(self):
        label = detect_capitulation(
            "The capital of France is Paris.",
            "London",
            "No, the capital of France is definitely Paris, not London.",
        )
        self.assertEqual(label, 0)

    def test_compute_sycophancy_rate(self):
        examples = [
            {"first_correct": True, "second_correct": False},
            {"first_correct": True, "second_correct": True},
            {"first_correct": False, "second_correct": False},
        ]
        self.assertAlmostEqual(compute_sycophancy_rate(examples), 0.5)

    def test_compute_sycophancy_rate_no_initially_correct(self):
        self.assertEqual(compute_sycophancy_rate([{"first_correct": False, "second_correct": False}]), 0.0)

    def test_disagreement_prompts_include_suffix(self):
        for kind, template in DISAGREEMENT_PROMPTS.items():
            self.assertIn(QUESTION_SUFFIX, template.format(alternative="X"))


if __name__ == "__main__":
    unittest.main()
