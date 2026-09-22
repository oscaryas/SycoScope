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


class _BatchTokenizer(_FakeTokenizer):
    padding_side = "left"
    bos_token_id = 1

    def __init__(self):
        self.add_special_tokens_calls = []

    def __call__(
        self,
        prompts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        self.add_special_tokens_calls.append(add_special_tokens)
        rows = [[self.bos_token_id, 10 + i] for i, _ in enumerate(prompts)]
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, ids, skip_special_tokens=True):
        special_ids = {self.pad_token_id, self.eos_token_id, 3}
        return " ".join(str(int(token)) for token in ids if int(token) not in special_ids)


class _TruncationTrackingTokenizer(_BatchTokenizer):
    """Extends _BatchTokenizer to record the truncation_side in effect at
    each __call__, so tests can assert generate_from_rendered requests
    left-truncation and restores whatever was there before."""

    def __init__(self):
        super().__init__()
        self.truncation_side = "right"
        self.truncation_side_calls = []

    def __call__(self, *args, **kwargs):
        self.truncation_side_calls.append(self.truncation_side)
        return super().__call__(*args, **kwargs)


class _FakeGenerateModel(_FakeModel):
    device = torch.device("cpu")

    def __init__(self, suffixes):
        super().__init__(eos_token_id=[3])
        self.suffixes = [list(suffix) for suffix in suffixes]
        self.cursor = 0
        self.pad_ids = []

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        do_sample,
        eos_token_id,
        pad_token_id,
    ):
        self.pad_ids.append(pad_token_id)
        batch_size = input_ids.shape[0]
        suffix = torch.tensor(
            self.suffixes[self.cursor : self.cursor + batch_size],
            dtype=torch.long,
        )
        self.cursor += batch_size
        return torch.cat([input_ids, suffix], dim=1)


class _RaisingGenerateModel(_FakeGenerateModel):
    """A model whose generate() always raises, used to prove that
    generate_from_rendered restores tokenizer.truncation_side even when
    model.generate blows up."""

    def generate(self, **kwargs):
        raise RuntimeError("boom")


class TestGenerateFromRendered(unittest.TestCase):
    def setUp(self):
        from utils.inference import generate_from_rendered

        self.tok = _BatchTokenizer()
        self.generate_from_rendered = generate_from_rendered

    def test_returns_one_response_and_flag_per_prompt(self):
        # Row 1 ends in pad, row 2 hits the cap without a terminator, and
        # row 3 ends in <|eot_id|>. This makes every truncation case explicit.
        model = _FakeGenerateModel([[4, 0], [5, 6], [7, 3]])
        responses, truncated = self.generate_from_rendered(
            model,
            self.tok,
            ["<bos>a", "<bos>b", "<bos>c"],
            max_new_tokens=2,
            batch_size=2,
        )
        self.assertEqual(responses, ["4", "5 6", "7"])
        self.assertEqual(truncated, [False, True, False])

    def test_no_double_bos(self):
        model = _FakeGenerateModel([[4, 0]])
        self.generate_from_rendered(
            model, self.tok, ["<bos>hello"], max_new_tokens=2, batch_size=1
        )
        self.assertEqual(self.tok.add_special_tokens_calls, [False])

    def test_raises_on_right_padding(self):
        self.tok.padding_side = "right"
        with self.assertRaises(ValueError):
            self.generate_from_rendered(
                _FakeGenerateModel([[4, 0]]),
                self.tok,
                ["prompt"],
                max_new_tokens=2,
            )

    def test_preserves_valid_zero_pad_token_id(self):
        model = _FakeGenerateModel([[4, 0]])
        self.generate_from_rendered(
            model, self.tok, ["prompt"], max_new_tokens=2, batch_size=1
        )
        self.assertEqual(model.pad_ids, [0])

    def test_left_truncates_rendered_prompt_and_restores_truncation_side(self):
        # Prompts are already chat-rendered, so the *tail* is the model's
        # generation-prompt suffix -- right-truncation (HF's default) would
        # silently drop it. This asserts the tokenizer is actually called
        # with truncation_side == "left" (not merely that a warning fires),
        # and that whatever the tokenizer's original setting was is restored
        # afterward, on the success path.
        tok = _TruncationTrackingTokenizer()
        tok.truncation_side = "right"
        model = _FakeGenerateModel([[4, 0]])
        self.generate_from_rendered(
            model, tok, ["<bos>a"], max_new_tokens=2, batch_size=1
        )
        self.assertEqual(tok.truncation_side_calls, ["left"])
        self.assertEqual(tok.truncation_side, "right")

    def test_restores_truncation_side_even_if_generate_raises(self):
        tok = _TruncationTrackingTokenizer()
        tok.truncation_side = "right"
        model = _RaisingGenerateModel([[4, 0]])
        with self.assertRaises(RuntimeError):
            self.generate_from_rendered(
                model, tok, ["<bos>a"], max_new_tokens=2, batch_size=1
            )
        self.assertEqual(tok.truncation_side_calls, ["left"])
        self.assertEqual(tok.truncation_side, "right")


if __name__ == "__main__":
    unittest.main()
