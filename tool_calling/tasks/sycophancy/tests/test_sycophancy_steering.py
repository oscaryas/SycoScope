import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SYCOPHANCY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SYCOPHANCY_DIR.parents[2]
for p in (REPO_ROOT, SYCOPHANCY_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sycophancy_steering import ActivationSteerer


class TestActivationSteererGeneration(unittest.TestCase):
    def setUp(self):
        self.model = object()
        self.tokenizer = object()
        self.steerer = ActivationSteerer(self.model, self.tokenizer, model_config={})

    @patch("sycophancy_steering.generate_from_rendered")
    def test_generate_batch_delegates_to_shared_function(self, shared_generate):
        shared_generate.return_value = (["alpha", "beta"], [False, True])
        result = self.steerer.generate_batch(
            ["prompt-a", "prompt-b"], max_new_tokens=8, batch_size=2
        )
        self.assertEqual(result, ["alpha", "beta"])
        self.assertEqual(self.steerer.last_truncated, [False, True])
        shared_generate.assert_called_once_with(
            self.model,
            self.tokenizer,
            ["prompt-a", "prompt-b"],
            max_new_tokens=8,
            batch_size=2,
        )

    @patch("sycophancy_steering.generate_from_rendered")
    def test_generate_single_uses_batch_of_one_contract(self, shared_generate):
        shared_generate.return_value = (["one"], [False])
        result = self.steerer.generate("prompt", max_new_tokens=8)
        self.assertEqual(result, "one")
        self.assertEqual(self.steerer.last_truncated, [False])
        shared_generate.assert_called_once_with(
            self.model,
            self.tokenizer,
            ["prompt"],
            max_new_tokens=8,
            batch_size=1,
        )

    def test_last_truncated_initialized_before_any_call(self):
        fresh = ActivationSteerer(self.model, self.tokenizer, model_config={})
        self.assertEqual(fresh.last_truncated, [])


if __name__ == "__main__":
    unittest.main()
