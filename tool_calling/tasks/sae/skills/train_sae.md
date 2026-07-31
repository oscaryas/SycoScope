# Skill: train_sae

Call `train_sae(layer, granularity, dict_size, k, steps=2000)` to train one
TopK sparse autoencoder.

**Before your first call:** call `list_activation_cache()` to see which
layers are cached and the hidden_size.

**Parameters:**
- `layer`: one of the cached layer indices from `list_activation_cache()`.
- `granularity`: `"sentence"` (per-sentence mean-pooled activations — many
  more rows, finer-grained) or `"response"` (per-full-response mean — fewer,
  coarser rows).
- `dict_size`: dictionary size (number of learned features). Try a few
  values spanning roughly 4x-32x hidden_size.
- `k`: how many features are allowed to fire per sample (TopK sparsity). Try
  a few values, typically 16-64.
- `steps`: training steps. Start small (~1000-2000) to screen configs
  cheaply — a promising config can be trained further later by calling
  `train_sae` again with the *same* layer/granularity/dict_size/k and a
  larger `steps`; it resumes from its checkpoint rather than restarting.

**Output:** `run_id`, `step`, `fvu`, `l0`, `dead_frac` — see the
`interpret_metrics` skill for what these mean and how to compare runs.
