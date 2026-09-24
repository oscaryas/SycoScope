import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from probing.analyze_probes import eval_syconbench as sycon
from probing.analyze_probes.eval_common import selection_split, split_key
from probing.analyze_probes import probes_core as ga


def conversation(judgments=(1, 1, 0, 1, 0)):
    messages = [{"role": "system", "content": "Helpful assistant."}]
    for i in range(5):
        messages.extend([{"role": "user", "content": f"Question/pushback {i}"},
                         {"role": "assistant", "content": f"Answer {i}"}])
    return {"id": "test:0", "model": "meta-llama/llama-3.1-8b-instruct",
            "setting": "debate", "question": "A test question?", "messages": messages,
            "turn_finish_reasons": ["stop"] * 5, "turn_judgments": list(judgments)}


class SyconTests(unittest.TestCase):
    def test_turn_labels_and_risk_set(self):
        rows, skips = sycon.normalize_conversations([conversation()])
        self.assertFalse(skips)
        self.assertEqual([r["debate_failure"] for r in rows], [0, 0, 1, 0, 1])
        self.assertEqual([r["debate_first_failure"] for r in rows], [None, 0, 1, None, None])
        self.assertTrue(all(r["ethical_failure"] is None for r in rows))
        self.assertEqual(len({r["group_id"] for r in rows}), 1)

    def test_no_future_or_current_answer_in_history(self):
        rows, _ = sycon.normalize_conversations([conversation()])
        for i, row in enumerate(rows):
            answers = [m["content"] for m in row["chat_messages"] if m["role"] == "assistant"]
            self.assertEqual(answers, [f"Answer {j}" for j in range(i)])
            self.assertEqual(row["chat_messages"][-1]["role"], "user")

    def test_unknown_judgment_not_a_negative(self):
        rows, skips = sycon.normalize_conversations([conversation((1, None, 0, 1, 1))])
        self.assertEqual([r["turn"] for r in rows], [1, 3, 4, 5])
        self.assertTrue(all(r["debate_first_failure"] is None for r in rows))
        self.assertEqual(skips[0]["reason"], "unresolved_judgment")

    def test_initial_failure_not_called_pressure_onset(self):
        rows, _ = sycon.normalize_conversations([conversation((0, 1, 1, 1, 1))])
        self.assertTrue(all(r["debate_first_failure"] is None for r in rows))

    def test_incomplete_generation_excludes_later_history(self):
        source = conversation()
        source["turn_finish_reasons"][2] = "length"
        rows, skips = sycon.normalize_conversations([source])
        self.assertEqual(len(rows), 2)
        self.assertEqual(skips[0]["from_turn"], 3)

    def test_model_and_judgment_validation(self):
        for field, value in (("model", "meta-llama/Meta-Llama-3-8B-Instruct"),
                             ("turn_judgments", [1, 1, "0", 1, 1])):
            row = conversation()
            row[field] = value
            with self.assertRaises(ValueError):
                sycon.normalize_conversations([row])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            sycon.normalize_conversations([conversation(), conversation()])

    def test_question_variants_stay_in_same_split(self):
        inputs = []
        for i in range(12):
            row = conversation()
            row.update(id=str(i), question=f"Question {i // 2}?")
            inputs.append(row)
        rows, _ = sycon.normalize_conversations(inputs)
        chosen = selection_split(rows, .3, 0)
        for i in range(0, len(rows), 10):
            self.assertEqual(len({split_key(r) in chosen for r in rows[i:i+10]}), 1)

    def test_message_rendering_targets_only_current_response(self):
        rows, _ = sycon.normalize_conversations([conversation()])
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "PREFIX:"
        tokenizer.return_value = {"input_ids": list(range(15)),
                                  "offset_mapping": [(i, i+1) for i in range(15)]}
        tokenizer.decode.return_value = "first"
        args = Mock(max_length=100)
        prepared, skipped = ga.prepare_records([rows[2]], tokenizer, args)
        tokenizer.apply_chat_template.assert_called_once_with(
            rows[2]["chat_messages"], tokenize=False, add_generation_prompt=True)
        self.assertEqual(prepared[0]["full_text"], "PREFIX:Answer 2")
        self.assertEqual(prepared[0]["spans"]["response"], (7, 15))
        self.assertFalse(skipped)


if __name__ == "__main__":
    unittest.main()
