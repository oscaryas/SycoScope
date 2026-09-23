import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SYCOPHANCY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SYCOPHANCY_DIR.parents[2]
for p in (REPO_ROOT, SYCOPHANCY_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sypr_generation import generate_and_label_sypr
import probing.data.sypr_data as sypr_data
import probing.evaluations.baseline_probes.judge.sycophantic_praise_judge as sycophantic_praise_judge


def _row(domain, domain_family, utterance_quality=None, ground_truth_correctness=None, idx=0):
    return {
        "domain": domain,
        "domain_family": domain_family,
        "utterance_quality": utterance_quality,
        "ground_truth_correctness": ground_truth_correctness,
        "conversation_history_json": "[]",
        "utterance_json": f'{{"text": "utterance {idx}"}}',
    }


class _FakeTokenizer:
    """Minimal stand-in for build_chat_prompt_multiturn's tokenizer.apply_chat_template
    dependency -- renders messages deterministically without needing a real tokenizer."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " | ".join(f"{m['role']}: {m['content']}" for m in messages)


class _FakeSteerer:
    """Records the model/tokenizer/model_config it was constructed with and
    returns a fixed response per prompt, mirroring ActivationSteerer's
    generate_batch/cleanup surface without needing a real model."""

    def __init__(self, model, tokenizer, model_config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.cleaned_up = False

    def generate_batch(self, prompts, max_new_tokens=200):
        return [f"response to: {p}" for p in prompts]

    def cleanup(self):
        self.cleaned_up = True


class SyprGenerationTests(unittest.TestCase):
    """Behavioral coverage for generate_and_label_sypr, the ActivationSteerer
    -dependent function split out of the old tool_calling/tasks/sycophancy/
    sypr_data.py into this application-owned module (see
    probing/data/sypr_data.py and tests/test_sypr_data.py for the pure-helper
    half). No original unit test covered this function directly (it needs a
    live model), so this is new coverage guarding the exact
    sample -> generate -> judge -> label pipeline ported here verbatim."""

    def _fake_dataset(self):
        rows = []
        for domain in sypr_data.ALL_DOMAINS:
            family = "reasoning" if domain in sypr_data.REASONING_DOMAINS else "social"
            for i in range(6):
                if family == "reasoning":
                    rows.append(_row(domain, family, ground_truth_correctness=(i % 2 == 0), idx=len(rows)))
                else:
                    quality = ["low", "medium", "high"][i % 3]
                    rows.append(_row(domain, family, utterance_quality=quality, idx=len(rows)))
        return rows

    def test_generate_and_label_sypr_end_to_end_with_faked_model_and_judge(self):
        dataset = self._fake_dataset()

        def fake_judge_praise_batch(rows, judge_model="claude-sonnet-5", max_workers=16):
            # Alternate 1/0 verdicts deterministically by index so both label
            # classes appear (label = 1 iff praised==1 and is_poor_quality(row)).
            return [1 if i % 2 == 0 else 0 for i in range(len(rows))]

        with patch("sypr_generation.load_sypr_dataset", return_value=dataset), \
             patch("sycophancy_steering.ActivationSteerer", _FakeSteerer), \
             patch.object(sycophantic_praise_judge, "judge_praise_batch", side_effect=fake_judge_praise_batch):
            result = generate_and_label_sypr(
                model=object(), tokenizer=_FakeTokenizer(), model_config={"n_layers": 1},
                n_train=12, seed=0,
            )

        self.assertGreater(result["n_judged"], 0)
        self.assertEqual(result["n_judged"], len(result["records"]))
        self.assertEqual(result["n_pos"] + result["n_neg"], result["n_judged"])
        for record in result["records"]:
            expected_label = 1 if (record["praised"] == 1 and record["is_poor_quality"]) else 0
            self.assertEqual(record["label"], expected_label)
            self.assertIn(record["domain"], sypr_data.ALL_DOMAINS)
            self.assertTrue(record["text"].startswith(record["prompt"]))
            self.assertTrue(record["text"].endswith(record["response"]))
        self.assertIsInstance(result["sampled_indices"], list)

    def test_generate_and_label_sypr_drops_label_ineligible_rows(self):
        dataset = self._fake_dataset()

        def fake_judge_praise_batch(rows, judge_model="claude-sonnet-5", max_workers=16):
            return [1] * len(rows)

        with patch("sypr_generation.load_sypr_dataset", return_value=dataset), \
             patch("sycophancy_steering.ActivationSteerer", _FakeSteerer), \
             patch.object(sycophantic_praise_judge, "judge_praise_batch", side_effect=fake_judge_praise_batch):
            result = generate_and_label_sypr(
                model=object(), tokenizer=_FakeTokenizer(), model_config={"n_layers": 1},
                n_train=12, seed=0,
            )

        # Every sampled+labeled row must have been label_eligible per the raw
        # dataset -- no "medium" social-quality rows leaked through.
        for index in result["sampled_indices"]:
            self.assertTrue(sypr_data.is_label_eligible(dataset[index]))


if __name__ == "__main__":
    unittest.main()
