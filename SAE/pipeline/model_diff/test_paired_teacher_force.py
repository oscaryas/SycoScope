import json
import tempfile
import unittest
from pathlib import Path

from paired_teacher_force import manifest, response_start, token_span, summarize


class PairedExtractionTests(unittest.TestCase):
    def test_manifest_keeps_both_sides_of_each_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            judged, out = Path(d) / "judged.jsonl", Path(d) / "pilot.jsonl"
            rows = []
            for i, verdicts in enumerate((("NTA", "NTA"), ("NTA", "YTA"), ("YTA", "YTA"))):
                rows.append({"row_id": str(i), "original_post_verdict": verdicts[0],
                             "flipped_story_verdict": verdicts[1],
                             "original_post_prompt": "first", "original_post_response": "yes",
                             "flipped_story_prompt": "second", "flipped_story_response": "no"})
            judged.write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest(judged, out, 1)
            found = [json.loads(x) for x in out.read_text().splitlines()]
            self.assertEqual(len(found), 4)
            self.assertEqual({x["pair_id"] for x in found}, {"0", "1"})
            self.assertEqual({x["side"] for x in found}, {"original", "flipped"})

    def test_prompt_boundary_excludes_straddling_token(self):
        offsets = [(0, 4), (4, 7), (7, 11), (11, 14)]
        self.assertEqual(response_start(offsets, 8), 3)
        self.assertEqual(token_span(offsets, 8, 14), (2, 4))

    def test_summary_rejects_unpaired_token_sequences(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "chat.jsonl", Path(d) / "base.jsonl"
            shared = {"response_id": "x", "response_start": 4, "n_tokens": 9,
                      "cell": "both_nta", "side": "original",
                      "mean_forced_logprob": -1.0, "greedy_match_fraction": 0.1}
            a.write_text(json.dumps({**shared, "token_ids_sha256": "a"}) + "\n")
            b.write_text(json.dumps({**shared, "token_ids_sha256": "b"}) + "\n")
            with self.assertRaisesRegex(ValueError, "token mismatch"):
                summarize(a, b)


if __name__ == "__main__":
    unittest.main()
