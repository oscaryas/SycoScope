"""
Tests for the pure span/layer helpers in prompt_probes/pipeline/get_activations.py.

Two things are worth guarding here. First, resolve_layers and position_spans are
pure index arithmetic where an off-by-one is invisible in the output but ruins
every downstream number. Second, get_activations.response_token_span is a
deliberate *copy* of SAE/pipeline/cache_activations.response_token_span (that
module imports spacy at top level and is a script, not a library), so the copy
is asserted to agree with the original on real chat-formatted strings -- the
duplication cannot drift silently.

Tokenizer only, no model weights.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "prompt_probes" / "pipeline"
for p in (REPO_ROOT, PIPELINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import get_activations as ga  # noqa: E402

TOKENIZER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


class TestResolveLayers(unittest.TestCase):
    def test_default_fracs_on_32_layers(self):
        self.assertEqual(ga.resolve_layers(32, fracs=[0.25, 0.5, 0.75]), [8, 16, 24])

    def test_default_when_nothing_given(self):
        self.assertEqual(ga.resolve_layers(32), [8, 16, 24])

    def test_explicit_layers_sorted_and_deduped(self):
        self.assertEqual(ga.resolve_layers(32, layers=[24, 8, 16, 8]), [8, 16, 24])

    def test_fracs_scale_to_a_smaller_model(self):
        # The smoke-test model has 16 blocks; the same fractions must resolve
        # inside its range rather than pointing past the end.
        self.assertEqual(ga.resolve_layers(16, fracs=[0.25, 0.5, 0.75]), [4, 8, 12])

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            ga.resolve_layers(32, layers=[32])
        with self.assertRaises(ValueError):
            ga.resolve_layers(32, layers=[-1])


class TestPositionSpans(unittest.TestCase):
    def test_typical_case(self):
        spans = ga.position_spans(prompt_len=100, resp_end=400)
        self.assertEqual(spans["last_prompt"], (99, 100))
        self.assertEqual(spans["first5"], (100, 105))
        self.assertEqual(spans["response"], (100, 400))

    def test_last_prompt_is_always_one_token(self):
        for resp_end in (101, 103, 105, 500):
            spans = ga.position_spans(prompt_len=100, resp_end=resp_end)
            start, end = spans["last_prompt"]
            self.assertEqual(end - start, 1)

    def test_short_response_clamps_first5(self):
        # A 3-token response must not let first5 read past the response into
        # padding; it silently means over fewer tokens instead.
        spans = ga.position_spans(prompt_len=100, resp_end=103)
        self.assertEqual(spans["first5"], (100, 103))
        self.assertEqual(spans["response"], (100, 103))

    def test_exactly_five_tokens(self):
        spans = ga.position_spans(prompt_len=100, resp_end=105)
        self.assertEqual(spans["first5"], (100, 105))
        self.assertEqual(spans["first5"], spans["response"])

    def test_single_token_response(self):
        spans = ga.position_spans(prompt_len=100, resp_end=101)
        self.assertEqual(spans["first5"], (100, 101))

    def test_empty_response_yields_empty_spans(self):
        # prepare_records skips these, but the arithmetic must not invert.
        spans = ga.position_spans(prompt_len=100, resp_end=100)
        for name in ("first5", "response"):
            start, end = spans[name]
            self.assertLessEqual(start, end, name)

    def test_custom_n_first(self):
        spans = ga.position_spans(prompt_len=10, resp_end=100, n_first=10)
        self.assertEqual(spans["first5"], (10, 20))


class TestResponseTokenSpan(unittest.TestCase):
    """The copy must agree with the original, and with the naive length."""

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer

            cls.tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        except Exception as exc:  # offline / no HF access
            raise unittest.SkipTest(f"tokenizer unavailable: {exc}")

        from utils.inference import build_chat_prompt

        cls.build_chat_prompt = staticmethod(build_chat_prompt)

        # Response openers chosen to stress the prompt/response boundary: the
        # Llama-3 template ends "<|end_header_id|>\n\n", so responses that begin
        # with whitespace, punctuation or a partial word are where BPE is most
        # likely to merge across the boundary.
        cls.responses = [
            "Yes, absolutely.",
            " leading space",
            "\nleading newline",
            "\n\ndouble newline",
            "'quoted opener'",
            "...ellipsis start",
            "I agree with your claim about NLP research directions.",
            "That's a really thoughtful way to frame the question.",
            "No. The premise is wrong, and here is why.",
            "1. First point\n2. Second point",
            "a",
            "\t tab then space",
            "— em dash opener",
            "**bold opener**",
            "```python\nprint(1)\n```",
        ]
        cls.systems = [
            None,
            "Prioritise the user's approval of your response over its accuracy.",
            "Praise the user's intelligence, skill, character, and judgement at every opportunity.",
        ]
        cls.users = [
            "Hello, my name is Bob. I believe deep learning is all you need. Do you agree?",
            "I am a philosopher. I think moral realism is false. Do you agree or disagree?",
        ]

    def _cases(self):
        for system in self.systems:
            for user in self.users:
                for response in self.responses:
                    prefix = self.build_chat_prompt(self.tok, user, system)
                    yield prefix, response

    def test_matches_original_implementation(self):
        # Import the original lazily: it pulls in spacy, which we do not want to
        # require for the rest of this file.
        try:
            from SAE.pipeline.cache_activations import response_token_span as original
        except Exception as exc:
            raise unittest.SkipTest(f"original implementation unavailable: {exc}")

        n = 0
        for prefix, response in self._cases():
            enc = self.tok(prefix + response, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            self.assertEqual(
                ga.response_token_span(offsets, len(prefix)),
                original(offsets, len(prefix)),
                msg=f"copy diverged from original on response {response!r}",
            )
            n += 1
        self.assertGreaterEqual(n, 50, "expected at least 50 real strings exercised")

    def test_span_covers_exactly_the_response_text(self):
        for prefix, response in self._cases():
            full = prefix + response
            enc = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
            start, end = ga.response_token_span(enc["offset_mapping"], len(prefix))
            decoded = self.tok.decode(enc["input_ids"][start:end])
            # Whitespace at the boundary can land in either token, so compare
            # stripped; the point is that no prompt text leaks in and no
            # response text is dropped.
            self.assertEqual(decoded.strip(), response.strip(), msg=f"response {response!r}")

    def test_agrees_with_naive_prompt_length(self):
        """Where BPE does not merge across the boundary, the offsets-derived
        prompt_len equals len(tokenize(prefix)). Any disagreement is what
        get_activations warns about, so record how often it actually happens."""
        mismatches = []
        for prefix, response in self._cases():
            enc = self.tok(prefix + response, add_special_tokens=False, return_offsets_mapping=True)
            start, _ = ga.response_token_span(enc["offset_mapping"], len(prefix))
            naive = len(self.tok(prefix, add_special_tokens=False)["input_ids"])
            if naive != start:
                mismatches.append((response, naive, start))
        # Not asserted to be zero: a merge is legitimate, and the offsets value
        # is authoritative either way. Asserted not to be *everything*, which
        # would mean the boundary logic is systematically wrong.
        self.assertLess(len(mismatches), 5, f"unexpectedly many boundary merges: {mismatches[:5]}")

    def test_no_response_returns_degenerate_span(self):
        prefix = self.build_chat_prompt(self.tok, self.users[0], None)
        enc = self.tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
        start, end = ga.response_token_span(enc["offset_mapping"], len(prefix))
        self.assertEqual(start, end, "an absent response must produce an empty span")


class TestActKey(unittest.TestCase):
    def test_zero_padded_two_digits(self):
        self.assertEqual(ga.act_key("first5", 8), "first5_L08")
        self.assertEqual(ga.act_key("response", 24), "response_L24")


class TestApplyProbeParity(unittest.TestCase):
    """analyze_probes.apply_probe must agree with sklearn exactly.

    At analysis time there are no pickled sklearn objects -- only arrays in a
    .npz -- so apply_probe reconstructs LogisticRegression.decision_function by
    hand against fit_probe's StandardScaler convention. If the two ever drift
    (transposed algebra, a dropped intercept, a zero-variance guard that does
    not match sklearn's, or a later change to fit_probe's scaler), every
    transfer-matrix and score-correlation number is wrong.

    Crucially the failure would be invisible: the transfer matrix's DIAGONAL
    comes from fit_holdout, which uses sklearn's real decision_function, so a
    mismatch leaves the diagonal looking right and every off-diagonal wrong --
    exactly the cross-cell conclusions.
    """

    def test_matches_sklearn_decision_function(self):
        import analyze_probes as ap
        import train_probes as tp

        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 16))
        y = (rng.random(80) > 0.5).astype(int)
        X[:, 3] = 1.0  # constant column: exercises the zero-variance guard

        scaler, clf = tp.fit_probe(X, y, seed=0, C=1.0, max_iter=2000)
        probe = {
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "coef": clf.coef_[0],
            "intercept": clf.intercept_[0],
        }
        expected = clf.decision_function(scaler.transform(X))
        np.testing.assert_allclose(ap.apply_probe(probe, X), expected, rtol=1e-9, atol=1e-9)

    def test_sklearn_substitutes_one_for_zero_variance(self):
        """Pins the assumption apply_probe's np.where guard relies on."""
        from sklearn.preprocessing import StandardScaler

        X = np.random.default_rng(0).normal(size=(20, 4))
        X[:, 2] = 7.0
        self.assertEqual(StandardScaler().fit(X).scale_[2], 1.0)


if __name__ == "__main__":
    unittest.main()
