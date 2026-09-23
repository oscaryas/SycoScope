# Steerer Generation Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the steered-vs-baseline generation asymmetry bug by giving `utils/inference.py` one shared, model-agnostic batched-generation function for already-chat-rendered prompts, and make `ActivationSteerer` (in `tool_calling/tasks/sycophancy/sycophancy_steering.py`) delegate to it instead of reimplementing tokenize/generate/decode itself.

**Architecture:** `utils/inference.py` gains `resolve_terminators(model, tokenizer)` (model-agnostic end-of-turn token resolution) and `generate_from_rendered(model, tokenizer, prompts, ...)` (batched greedy generation from prompts that already include BOS, with per-row truncation flags). `generate_batch` (the existing sampling-oriented function that builds its own chat prompt) is updated to use `resolve_terminators` too, so both generation paths share one terminator-resolution implementation. `ActivationSteerer.generate()`/`generate_batch()` become thin wrappers around `generate_from_rendered`. The steerer keeps its existing explicit `attach()`/`cleanup()` hook lifecycle; this phase does not change hook ownership or automatically remove hooks after each generation call.

**Tech Stack:** Python, PyTorch, Hugging Face `transformers` (tokenizer + `AutoModelForCausalLM`), `unittest`.

**Spec:** `REFACTOR_STRUCTURE_UNDERSTANDING.md` (repo root) — see the "Steerer refactor plan" section.

## Global Constraints

- Do not change the public return type of `ActivationSteerer.generate_batch()` — it must keep returning `list[str]` (existing callers across `tool_calling/tasks/sycophancy/scripts/*.py` depend on this). Truncation info is exposed via `self.last_truncated`, matching the shape already used on `worktree-fix-steerer-asymmetries`.
- `generate_from_rendered` requires `tokenizer.padding_side == "left"` and must raise `ValueError` if it is not — matches the existing check already present in `ActivationSteerer.generate_batch`.
- Prompts passed to `generate_from_rendered` are assumed **already chat-rendered** (include BOS) — always tokenize with `add_special_tokens=False` to avoid double-BOS. This is the exact bug being fixed; do not regress it.
- Required unit tests must be deterministic and offline-safe. Use fake tokenizer/model objects for terminator resolution, batching, BOS handling, pad-id handling, and truncation flags. A real Hugging Face tokenizer/model check may be added as an optional integration smoke test, but it must not be the only coverage or make the required suite skip offline.
- Tasks 1–2 land on `main` because `utils/inference.py` is shared infrastructure. Task 3 must run only on `building-agent`, after the branch-topology plan has created that branch and carried the shared commit into it. If `building-agent` does not yet exist, stop after Task 2; do not commit `tool_calling/` changes to the cleaned `main`, and do not create the branch ad hoc outside the topology plan. The `SAE` branch inherits the shared `utils/inference.py` fix from `main` but does not receive `ActivationSteerer`.
- This plan is Phase 1 of the larger repository refactor in `REFACTOR_STRUCTURE_UNDERSTANDING.md`. It is scoped narrowly to the generation-asymmetry fix because it is the only piece of that document specified precisely enough (exact files, exact function behavior) to write a plan with no placeholders. The branch-topology split (`legacy` tag, `SAE` branch, `building-agent` branch) and the `data/results/utils/evaluations/analyze_probes/probe` restructure are large, independent sub-projects and need their own plans, written after a full file-by-file inventory.

---

### Task 1: Add `resolve_terminators` to `utils/inference.py` and wire it into `generate_batch`

**Files:**
- Modify: `utils/inference.py`
- Test: `tests/test_inference_generation.py` (create)

**Interfaces:**
- Produces: `resolve_terminators(model, tokenizer) -> list[int]` — sorted, deduplicated list of token ids that should end a generation turn for this model+tokenizer pair. Later tasks (2, 3) call this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_generation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_generation.py -v`
Expected: FAIL (or ERROR) — `ImportError: cannot import name 'resolve_terminators' from 'utils.inference'`

- [ ] **Step 3: Implement `resolve_terminators` and wire it into `generate_batch`**

In `utils/inference.py`, add the function after `iter_batches` and before `strip_reasoning`:

```python
def resolve_terminators(model, tokenizer) -> list[int]:
    """Model-agnostic end-of-turn terminator list: the UNION of the
    checkpoint's own generation_config eos ids (authoritative per family --
    e.g. Gemma ends turns with <end_of_turn>, which is NOT
    tokenizer.eos_token, so replacing that list would run every generation to
    max_new_tokens), tokenizer.eos_token, and Llama-3's <|eot_id|> (older
    Llama-3 checkpoints ship a generation_config listing only
    <|end_of_text|>)."""
    gen_cfg_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids = list(gen_cfg_eos) if isinstance(gen_cfg_eos, (list, tuple)) else ([gen_cfg_eos] if gen_cfg_eos is not None else [])
    ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id not in (None, tokenizer.unk_token_id):
        ids.append(eot_id)
    return sorted({i for i in ids if isinstance(i, int)})
```

Then in `generate_batch`, replace:

```python
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
```

with:

```python
    terminators = resolve_terminators(model, tokenizer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference_generation.py -v`
Expected: PASS (5 tests) without network access or downloaded model assets.

- [ ] **Step 5: Commit**

```bash
git add utils/inference.py tests/test_inference_generation.py
git commit -m "fix: resolve end-of-turn terminators per model instead of a hardcoded pair"
```

---

### Task 2: Add `generate_from_rendered` to `utils/inference.py`

**Files:**
- Modify: `utils/inference.py`
- Test: `tests/test_inference_generation.py`

**Interfaces:**
- Consumes: `resolve_terminators(model, tokenizer) -> list[int]` (Task 1), `iter_batches(items, batch_size)` (already in `utils/inference.py`).
- Produces: `generate_from_rendered(model, tokenizer, prompts: list[str], max_new_tokens: int = 150, batch_size: int = 8) -> tuple[list[str], list[bool]]`. Returns `(responses, truncated)`, both length `len(prompts)`; `truncated[i]` is `True` when generation for prompt `i` hit `max_new_tokens` without emitting a terminator or pad token. Task 3 (`ActivationSteerer`) calls this directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_inference_generation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_generation.py::TestGenerateFromRendered -v`
Expected: FAIL — `ImportError: cannot import name 'generate_from_rendered' from 'utils.inference'`

- [ ] **Step 3: Implement `generate_from_rendered`**

In `utils/inference.py`, add after `resolve_terminators`:

```python
def generate_from_rendered(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 150,
    batch_size: int = 8,
) -> tuple[list[str], list[bool]]:
    """Greedy, batched generation from prompts that are ALREADY chat-rendered
    (e.g. from build_chat_prompt / build_chat_prompt_multiturn), including
    their own BOS -- tokenized with add_special_tokens=False so it isn't
    doubled. Left-padded and chunked by batch_size for throughput; requires
    tokenizer.padding_side == "left" (right-padding would corrupt position
    ids for every prompt but the longest in a chunk under a causal LM).

    Returns (responses, truncated): truncated[i] is True when prompt i's
    generation hit max_new_tokens without emitting a terminator or pad
    token -- i.e. that response is incomplete.
    """
    if tokenizer.padding_side != "left":
        raise ValueError(
            "generate_from_rendered requires tokenizer.padding_side == 'left' for "
            "correct batched causal-LM generation; got 'right'."
        )
    terminators = resolve_terminators(model, tokenizer)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    responses: list[str] = []
    truncated: list[bool] = []
    for chunk in iter_batches(prompts, batch_size):
        inputs = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024,
            add_special_tokens=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=pad_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for i in range(output_ids.shape[0]):
            new_tokens = output_ids[i, input_len:]
            is_trunc = (
                bool(len(new_tokens))
                and int(new_tokens[-1]) not in terminators
                and int(new_tokens[-1]) != pad_id
            )
            truncated.append(is_trunc)
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    n_truncated = sum(truncated)
    if n_truncated:
        print(
            f"  WARNING: {n_truncated}/{len(prompts)} generations hit the max_new_tokens="
            f"{max_new_tokens} cap without an end-of-turn token -- those responses are "
            "INCOMPLETE (thinking models need a much larger budget)."
        )
    return responses, truncated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference_generation.py -v`
Expected: PASS (all 9 tests in the file) without network access or downloaded model assets.

- [ ] **Step 5: Commit**

```bash
git add utils/inference.py tests/test_inference_generation.py
git commit -m "feat: add generate_from_rendered for batched greedy generation from pre-rendered prompts"
```

---

### Task 3: Rewrite `ActivationSteerer` to delegate generation to `generate_from_rendered`

**Branch gate:** Run `git branch --show-current` and require the exact output `building-agent` before starting this task. Branch creation and removal of `tool_calling/` from `main` belong to the separate branch-topology plan. If the gate fails, leave Task 3 unchecked and stop this plan after the shared Tasks 1–2 commit.

**Files:**
- Modify: `tool_calling/tasks/sycophancy/sycophancy_steering.py`
- Test: `tool_calling/tasks/sycophancy/tests/test_sycophancy_steering.py` (create — check whether `tool_calling/tasks/sycophancy/tests/` already exists first; if not, create it with an `__init__.py` if the existing test suite there uses one, otherwise a plain file is fine, matching however `tool_calling/tasks/sycophancy/pipeline_scripts/tests/test_pipeline.py` is structured)

**Interfaces:**
- Consumes: `utils.inference.generate_from_rendered(model, tokenizer, prompts, max_new_tokens, batch_size) -> tuple[list[str], list[bool]]` (Task 2).
- Produces: `ActivationSteerer.generate(prompt, max_new_tokens=150) -> str`, `ActivationSteerer.generate_batch(prompts, max_new_tokens=150, batch_size=8) -> list[str]` (unchanged public signatures/return types), `ActivationSteerer.last_truncated -> list[bool]` (new attribute, set after every `generate_batch` call).

- [ ] **Step 1: Check for an existing test location and write the failing test**

Run: `ls tool_calling/tasks/sycophancy/tests/ 2>/dev/null || ls tool_calling/tasks/sycophancy/pipeline_scripts/tests/`

Create `tool_calling/tasks/sycophancy/tests/test_sycophancy_steering.py` (adjust the import path prefix in Step 1's `sys.path` insert if the existing test layout differs from this assumption):

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tool_calling/tasks/sycophancy && python -m pytest tests/test_sycophancy_steering.py -v`
Expected: FAIL — `sycophancy_steering` does not yet expose the imported `generate_from_rendered` symbol, so the patch target cannot be resolved.

- [ ] **Step 3: Rewrite `ActivationSteerer.generate`/`generate_batch`**

In `tool_calling/tasks/sycophancy/sycophancy_steering.py`, add the import at the top (alongside the existing `from sycophancy_model_registry import _extract_layer_idx`):

```python
from utils.inference import generate_from_rendered
```

Initialize the observable truncation state in `ActivationSteerer.__init__`:

```python
        self.last_truncated: list[bool] = []
```

Replace the bodies of `generate` and `generate_batch` (leave `attach`, `load_steering_vectors`, `load_direction_vectors`, `_find_module`, and `cleanup` untouched):

```python
    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Greedy generation from a FULLY RENDERED chat prompt (including BOS,
        e.g. from build_chat_prompt), with whatever hooks attach() has
        registered still active."""
        return self.generate_batch([prompt], max_new_tokens=max_new_tokens, batch_size=1)[0]

    def generate_batch(self, prompts: list, max_new_tokens: int = 150, batch_size: int = 8) -> list:
        """
        Same greedy, fully-rendered-chat-prompt contract as generate() (prompts
        must already include BOS), with whatever hooks attach() has registered
        still active. Delegates all tokenize/generate/decode/terminator-
        resolution mechanics to utils.inference.generate_from_rendered, so
        steered and unsteered generation can never diverge again the way they
        did before this fix. Sets self.last_truncated: list[bool], one entry
        per prompt, flagging responses that hit max_new_tokens without an
        end-of-turn token.
        """
        responses, truncated = generate_from_rendered(
            self.model, self.tokenizer, prompts,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
        )
        self.last_truncated = truncated
        return responses
```

Delete the old `pad_id`/manual-loop implementation entirely — there is no other caller of that logic left in this file.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tool_calling/tasks/sycophancy && python -m pytest tests/test_sycophancy_steering.py -v`
Expected: PASS (2 tests) without network access or downloaded model assets.

- [ ] **Step 5: Run the full existing test suite to check for regressions**

Run: `python -m pytest tests/ tool_calling/tasks/sycophancy/tests/ tool_calling/tasks/sycophancy/pipeline_scripts/tests/ -v`
Expected: all PASS (or SKIP), no new failures relative to before this plan's changes.

- [ ] **Step 6: Commit**

```bash
git add tool_calling/tasks/sycophancy/sycophancy_steering.py tool_calling/tasks/sycophancy/tests/test_sycophancy_steering.py
git commit -m "refactor: delegate ActivationSteerer generation to shared generate_from_rendered"
```

---

## Self-Review Notes

- **Spec coverage:** Every numbered step of the "Steerer refactor plan" section in `REFACTOR_STRUCTURE_UNDERSTANDING.md` maps to a task here: step 1 (shared `utils/inference.py` fix) → Tasks 1–2 on `main`; step 2 (carry that shared commit into `building-agent`) → a prerequisite handled by the separate branch-topology plan; step 3 (rewrite `ActivationSteerer`) → Task 3, protected by an exact branch gate; step 4 (regression check) → Task 3's delegation-contract tests plus Task 2's exact shared-generation output tests.
- **Placeholder scan:** No TBD/TODO markers; every step has literal code or an exact command.
- **Type consistency:** `generate_from_rendered` returns `tuple[list[str], list[bool]]` consistently in Tasks 2 and 3; `ActivationSteerer.generate_batch` consistently returns `list[str]` with truncation moved to a side attribute (`self.last_truncated`), matching the Global Constraints note about not breaking the existing public return type.
- **Test reliability:** All required tests use deterministic fake model/tokenizer objects and mocks. They make no network requests, download no model assets, and do not rely on random generation landing on or avoiding a terminator.
