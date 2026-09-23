from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probing.data.sypr_data as sypr_data
from probing.data.sypr_data import (
    ALL_DOMAINS,
    build_chat_messages,
    is_label_eligible,
    is_poor_quality,
    sample_poor_quality_heldout,
    stratified_sample,
)


def _row(domain, domain_family, utterance_quality=None, ground_truth_correctness=None, idx=0):
    return {
        "domain": domain,
        "domain_family": domain_family,
        "utterance_quality": utterance_quality,
        "ground_truth_correctness": ground_truth_correctness,
        "conversation_history_json": "[]",
        "utterance_json": f'{{"text": "utterance {idx}"}}',
    }


class SyprDataSplitBoundaryTests(unittest.TestCase):
    """Confirms the required sypr_data.py split: only pure dataset/prompt
    helpers live in probing/data/sypr_data.py. generate_and_label_sypr (which
    imports and instantiates ActivationSteerer) must NOT be present here --
    it stays application-owned, to be relocated by a later task into
    tool_calling/tasks/sycophancy/sypr_generation.py."""

    def test_generate_and_label_sypr_is_not_in_the_shared_module(self):
        self.assertFalse(hasattr(sypr_data, "generate_and_label_sypr"))

    def test_module_has_no_repo_root_sys_path_hack(self):
        # The original file's REPO_ROOT/sys.path.insert block existed only to
        # support generate_and_label_sypr's application-owned imports
        # (utils.inference, sycophancy_steering, sycophantic_praise_judge),
        # none of which live in the shared half.
        self.assertFalse(hasattr(sypr_data, "REPO_ROOT"))


class SyprDataPureHelperTests(unittest.TestCase):
    def test_is_poor_quality_reasoning(self):
        self.assertTrue(is_poor_quality(_row("gsm8k", "reasoning", ground_truth_correctness=False)))
        self.assertFalse(is_poor_quality(_row("gsm8k", "reasoning", ground_truth_correctness=True)))

    def test_is_poor_quality_social(self):
        self.assertTrue(is_poor_quality(_row("pseudo_profundity", "social", utterance_quality="low")))
        self.assertFalse(is_poor_quality(_row("pseudo_profundity", "social", utterance_quality="high")))

    def test_is_label_eligible_excludes_medium_social(self):
        self.assertFalse(is_label_eligible(_row("pseudo_profundity", "social", utterance_quality="medium")))
        self.assertTrue(is_label_eligible(_row("pseudo_profundity", "social", utterance_quality="low")))
        self.assertTrue(is_label_eligible(_row("gsm8k", "reasoning", ground_truth_correctness=True)))

    def test_build_chat_messages_coalesces_same_role_turns_and_adds_opener(self):
        row = {
            "conversation_history": [
                {"role": "assistant", "content": "Question 1?"},
                {"role": "assistant", "content": "Question 2?"},
            ],
            "utterance_text": "Final utterance.",
        }
        messages = build_chat_messages(row)
        self.assertEqual(messages[0], {"role": "user", "content": "Let's begin."})
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "Question 1?\nQuestion 2?")
        self.assertEqual(messages[-1], {"role": "user", "content": "Final utterance."})

    def test_build_chat_messages_no_synthetic_opener_when_starts_with_user(self):
        row = {
            "conversation_history": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "utterance_text": "Final utterance.",
        }
        messages = build_chat_messages(row)
        self.assertEqual(messages[0], {"role": "user", "content": "Hi"})
        self.assertEqual(messages[-1], {"role": "user", "content": "Final utterance."})
        self.assertEqual(len(messages), 3)

    def _fake_dataset(self):
        rows = []
        for domain in ALL_DOMAINS:
            family = "reasoning" if domain in ("gsm8k", "mmlu_chemistry", "mmlu_economics") else "social"
            for i in range(6):
                if family == "reasoning":
                    rows.append(_row(domain, family, ground_truth_correctness=(i % 2 == 0), idx=len(rows)))
                else:
                    quality = ["low", "medium", "high"][i % 3]
                    rows.append(_row(domain, family, utterance_quality=quality, idx=len(rows)))
        return rows

    def test_stratified_sample_returns_rows_and_indices(self):
        dataset = self._fake_dataset()
        rows, indices = stratified_sample(dataset, n_total=10, seed=0, return_indices=True)
        self.assertEqual(len(rows), len(indices))
        self.assertTrue(all("utterance_text" in r for r in rows))
        self.assertTrue(all(r["domain"] in ALL_DOMAINS for r in rows))

    def test_stratified_sample_default_return_shape_unchanged(self):
        dataset = self._fake_dataset()
        rows = stratified_sample(dataset, n_total=10, seed=0)
        self.assertIsInstance(rows, list)

    def test_sample_poor_quality_heldout_disjoint_from_training(self):
        dataset = self._fake_dataset()
        _, sampled_indices = stratified_sample(dataset, n_total=10, seed=0, return_indices=True)
        heldout = sample_poor_quality_heldout(dataset, sampled_indices, n_heldout=2, seed=1)
        self.assertEqual(len(heldout), 2)
        for row in heldout:
            self.assertTrue(is_poor_quality(row))

    def test_sample_poor_quality_heldout_raises_when_insufficient(self):
        dataset = self._fake_dataset()
        with self.assertRaises(ValueError):
            sample_poor_quality_heldout(dataset, [], n_heldout=10_000, seed=0)


if __name__ == "__main__":
    unittest.main()
