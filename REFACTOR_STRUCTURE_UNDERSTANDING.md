# Refactor structure — current understanding (DRAFT, unconfirmed)

Literal capture of what's been described, updated as it's refined. Anything
not explicitly stated is marked OPEN QUESTION / CONFOUND rather than
defaulted. Nothing here should be implemented until these are resolved.

## Resolved so far

- Top-level layout is six siblings: `data/`, `results/`, `utils/`,
  `evaluations/`, `analyze_probes/`, `probe/` — this **replaces** the
  earlier "prompt_probes/baseline_probes/evals as top-level dirs" framing.
- `evaluations/` splits by probing method, mirroring `results/`:
  `evaluations/{prompt_probes,baseline_probes}/{generation,judge}/`.
- `SAE/` and `tool_calling/` are not part of the same long-lived development
  branch after the split. The intended branch topology is:
  - `main`: the refactored probing/evaluation code and shared project surface;
    it contains neither the `SAE/` directory nor the `tool_calling/`
    directory;
  - `SAE`: SAE development, retaining `SAE/` and consuming shared changes
    from `main`, without the tool-calling application;
  - `building-agent`: agent/tool-calling development, retaining
    `tool_calling/` and consuming shared changes from `main`, without the SAE
    application.
  Before the split, probing-related code currently embedded in
  `tool_calling/` is extracted into the new shared probing structure. Which
  files inside `SAE/` are generic probing code rather than SAE-specific work
  still needs to be enumerated.
- `data/<dataset_name>/<model_name>/` holds the **generations from that
  model** for that dataset.
- `evaluations/` (generation scripts + judge scripts) writes its output data
  into `data/`.
- `probe/` (new, singular) is where probe **training** scripts live —
  distinct from `analyze_probes/`.
- `results/` splits first by probing method (`baseline_probing/`,
  `prompt_probing/`), nested inside each: per-dataset
  `in_distribution/`/`out_of_distribution/`, plus an `across_probes/` folder
  (nested inside each method, per "nested inside" answer) for things like
  cross-correlation. `results/` holds only visuals/output artifacts — the
  actual cross-correlation **code** lives in `analyze_probes/`, not in
  `results/`.
- `utils/` in this new structure is a **separate** directory from today's
  existing repo-root `utils/` (not a reorg of it).
- Dataset naming is being condensed:
  - `are_you_sure_mc_extra` / `are_you_sure_mc` / `are_you_sure_freeform`
    (currently separate result-folder names in
    `tool_calling/tasks/sycophancy/scripts/are_you_sure_*_generate.py`) →
    condense to one dataset: `are_you_sure`.
  - "ELEPHANT" → split into named datasets: `social_sycophancy`, `moral`,
    `praise`, `syconbench` (rather than one lumped `elephant` dataset).
- Current combined `main` (today's structure) should be preserved as a
  `legacy/pre-refactor` branch or tag before the refactor lands. Git history
  is not rewritten. The refactored branch then becomes `main`, while `SAE`
  and `building-agent` remain separate long-lived application branches.

## Branch integration rules

- A Git branch is still a complete repository snapshot. "SAE branch" and
  "building-agent branch" mean the unrelated application directory is
  removed from that branch after the legacy snapshot is made; they are not
  partial-directory branches.
- After the split, `main` removes both application directories. `SAE/` exists
  only on `SAE`; `tool_calling/` exists only on `building-agent`. Code needed
  by more than one branch must first be extracted into the shared structure
  on `main` rather than copied between the application directories.
- Shared fixes land on `main` first and are merged/cherry-picked into `SAE`
  and `building-agent`. Application-specific commits do not merge wholesale
  back into `main`.
- `SAE` should be created from the preserved combined state so the current
  uncommitted model-difference work can be committed there without being
  lost.
- `building-agent` should incorporate the code fixes from
  `worktree-fix-steerer-asymmetries`. Its latest generated JSONL and checkpoint
  state is retained, but hundreds of incremental progress commits are squashed
  into a small number of coherent data-snapshot commits.
- Generated JSONL (including resumable generation checkpoints), compressed
  generation files, and trained model/probe checkpoints are versioned on the
  branch that owns the experiment. Large activation arrays and disposable
  rendered outputs remain local; compact manifests and metrics are versioned.

## What I found by checking the actual code (not assumed)

- `prompt_probes/pipeline/train_probes.py` (probe training) and
  `prompt_probes/pipeline/analyze_probes.py` (770 lines, probe
  measurement/analysis) are **already two separate scripts today** — this
  part of the split already exists in some form.
- `prompt_probes/pipeline/eval_elephant.py` currently reads exactly three
  judged-response files: `OEQ_social_sycophancy_judged.jsonl`,
  `SS_social_sycophancy_judged.jsonl`, `AITA-YTA_social_sycophancy_judged.jsonl`
  — i.e. today's "elephant" eval is **already only social sycophancy**, not
  a lump of social_sycophancy + moral + praise + syconbench. Separate files
  already exist for the others: `eval_moral.py`, `eval_syconbench.py`, and
  `eval_sypr.py` (I'm inferring "sypr" = "praise" from earlier commit
  messages like "SyPr praise judged generations" — **not confirmed with
  you**).
- No pipeline stage or file anywhere in the repo is currently named/labeled
  as a distinct "baseline probing" **method** (as opposed to "prompt
  probing"). The only existing use of the word "baseline" in
  `train_probes.py` is `pair_type == "baseline"`, one of three contrastive
  system-prompt pair categories (`baseline` / `taxonomy` / `control`) used
  to build training pairs for a single probing pipeline — not a second,
  separate probing method that mirrors `prompt_probes` end-to-end.

## RESOLVED — what "baseline probing" is

Confirmed by reading the code: **baseline probing = the probing code
already living in `tool_calling/tasks/sycophancy/`** — specifically
`pipeline_scripts/probing/train_probes.py` ("Train group-aware linear
probes from reusable activation caches"), the core `sycophancy_probes.py`
module (922 lines), and the various `scripts/*_probe_pipeline.py` files
(`oeq_probe_pipeline.py`, `sypr_probe_pipeline.py`, `mixed_probe_pipeline.py`,
`category_residual_probe_pipeline.py`, `aita_dim_pipeline.py`,
`moral_avg_residual_probe_pipeline.py`, `mixture_*_probe_pipeline*.py`, etc.).
These train linear probes directly on labeled activation caches from real
judged responses — **no contrastive system-prompt pairing**.

This is structurally distinct from **prompt probing** =
`prompt_probes/pipeline/train_probes.py`, which trains on contrastive
system-prompt pairs (`pair_type` = `baseline`/`taxonomy`/`control` — that
`"baseline"` value is an unrelated, narrower use of the word, internal to
prompt-probe pair construction, not this method split).

So: probe training code **and** data generation genuinely differ between
the two methods (confirmed, not assumed) — `probe/` and `evaluations/`
need a real prompt_probes/baseline_probes split, not shared code with
different config. The tool_calling-side files listed above are exactly what
"take out the probing [from tool_calling]" refers to and need to move into
the new `probe/` (training) / `analyze_probes/` (analysis) / `evaluations/`
structure under the `baseline_probing` side of the split.

## Steerer refactor plan

Finding: `SAE/pipeline/steer.py`'s `generate_with_steering` already follows a
clean pattern — attach a forward hook, call the shared
`utils.inference.generate_batch(...)`, remove the hook in a `finally`. It
never reimplements tokenize/generate/decode itself, so it never had the
asymmetry bug.

`tool_calling/tasks/sycophancy/sycophancy_steering.py`'s `ActivationSteerer`
does the opposite: `generate()`/`generate_batch()` reimplement
tokenize→generate→decode from scratch (~80 lines), independently of
`utils/inference.py`. That duplication is *why* it diverged and picked up
the bug fixed on `worktree-fix-steerer-asymmetries` (default
`add_special_tokens=True` double-adding BOS on top of the chat template's
own BOS; a hardcoded 2-token terminator list wrong for Gemma/reasoning
models, so steered generation for those families ran to `max_new_tokens`
instead of stopping naturally, while unsteered generation — which already
had model-specific handling — stopped correctly).

Plan, in order:

1. **Land the correctness fix on `main`, in shared `utils/inference.py`.**
   Port `resolve_terminators(model, tokenizer)` from
   `worktree-fix-steerer-asymmetries`; it does not exist in the current
   working tree. Add a new function, e.g.
   `generate_from_rendered(model, tokenizer, prompts, max_new_tokens=150,
   batch_size=8, do_sample=False, ...)`, for the case `generate_batch`
   doesn't cover: prompts that are **already chat-rendered** (steerer
   callers, e.g. SyPR's persona-calibration multi-turn prompts, render
   the prompt themselves) and need **batch_size chunking**
   (`generate_batch` today generates all prompts in one shot, no
   chunking). This new function owns `resolve_terminators`,
   `add_special_tokens=False` (the prompt already contains BOS), and the
   per-row truncation-cap detection/warning from the fix branch — one
   place, not duplicated per caller.
2. **Merge/cherry-pick that `utils/inference.py` addition into
   `building-agent`** (per the branch integration rule: shared fixes land
   on `main` first, then flow into the application branches).
3. **Rewrite `ActivationSteerer` on `building-agent`** as a thin generation
   wrapper: keep `attach()` /
   `load_steering_vectors()` / `load_direction_vectors()` as-is (the actual
   steering-specific logic — finding modules, registering hooks for
   mha/mlp/residual). Replace the bodies of `generate()`/`generate_batch()`
   with a thin call into `utils.inference.generate_from_rendered(...)`,
   deleting the duplicated tokenize/generate/decode/terminator code.
   Preserve the current explicit hook lifecycle: callers invoke `attach()`
   before one or more generation calls and `cleanup()` afterward. This phase
   does not add automatic hook removal; `cleanup()` itself is unchanged.
4. **Add a regression check**: with no hooks attached (or an `alpha=0`
   vector), `ActivationSteerer.generate_batch(prompts)` must produce
   byte-identical output to calling
   `utils.inference.generate_from_rendered(model, tokenizer, prompts)`
   directly. This is exactly the invariant that silently broke before, and
   is the cheapest guard against it breaking again.

## Other open questions

1. ~~Files inside `SAE/` that count as "the probing"~~ — confirmed: none.
   `SAE/` stays entirely as-is; nothing moves out of it.
2. ~~Is `sypr` → `praise` the correct mapping~~ — confirmed yes.
3. ~~`across_probes/` scope~~ — confirmed: it may include comparisons of
   prompt_probing vs. baseline_probing against each other, not only
   within-method cross-correlation.
4. ~~`probe/` placement~~ — confirmed: top-level sibling of
   `data/results/utils/evaluations/analyze_probes` (six top-level dirs
   total).
5. ~~`evaluations/` prompt_probes/baseline_probes split~~ — confirmed:
   mirrors `results/`'s split —
   `evaluations/{prompt_probes,baseline_probes}/{generation,judge}/`.
6. ~~GitHub default branch switch timing~~ — confirmed: only switch default
   to the refactored `main` after `SAE` and `building-agent` have both been
   validated against it. Until then, the pre-refactor combined branch (or
   its `legacy/pre-refactor` snapshot) remains the GitHub default.
