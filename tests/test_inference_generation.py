import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class _FakeGenerationConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class _FakeModel:
    def __init__(self, eos_token_id):
        self.generation_config = _FakeGenerationConfig(eos_token_id)


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 99

    def convert_tokens_to_ids(self, token):
        return 3 if token == "<|eot_id|>" else self.unk_token_id


class TestResolveTerminators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils.inference import resolve_terminators

        cls.tok = _FakeTokenizer()
        cls.resolve_terminators = staticmethod(resolve_terminators)

    def test_includes_tokenizer_eos(self):
        model = _FakeModel(eos_token_id=None)
        ids = self.resolve_terminators(model, self.tok)
        self.assertIn(self.tok.eos_token_id, ids)

    def test_includes_eot_id_for_llama3_template(self):
        model = _FakeModel(eos_token_id=None)
        ids = self.resolve_terminators(model, self.tok)
        eot_id = self.tok.convert_tokens_to_ids("<|eot_id|>")
        self.assertIn(eot_id, ids)

    def test_includes_generation_config_eos_list(self):
        # Gemma-style: generation_config.eos_token_id is a list distinct from
        # tokenizer.eos_token_id (e.g. <end_of_turn>), and must be unioned in,
        # not replaced.
        gemma_style_id = 99999
        model = _FakeModel(eos_token_id=[gemma_style_id])
        ids = self.resolve_terminators(model, self.tok)
        self.assertIn(gemma_style_id, ids)
        self.assertIn(self.tok.eos_token_id, ids)

    def test_includes_generation_config_eos_scalar(self):
        model = _FakeModel(eos_token_id=self.tok.eos_token_id)
        ids = self.resolve_terminators(model, self.tok)
        self.assertEqual(ids.count(self.tok.eos_token_id), 1)

    def test_result_is_sorted_and_deduped(self):
        model = _FakeModel(eos_token_id=[self.tok.eos_token_id, self.tok.eos_token_id])
        ids = self.resolve_terminators(model, self.tok)
        self.assertEqual(ids, sorted(set(ids)))


if __name__ == "__main__":
    unittest.main()
