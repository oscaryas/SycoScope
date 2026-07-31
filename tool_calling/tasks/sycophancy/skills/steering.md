# Skill: steering

Once probes are trained, their directions can be used to steer generation —
adding a probe's direction to a layer's activations and seeing whether it
shifts the model toward or away from sycophantic responses.

**Prerequisites:** `train_probe_family` for the component you want to steer
with, then `write_metrics` (this is what actually saves the direction
vectors to `final_probe/`, not `train_probe_family` alone).

**Workflow:**
1. Call `list_steering_vectors()` to see which `(component, key)` pairs are
   available. Keys are `"(layer, head)"` for `mha`, or `layer` for `mlp`/
   `residual`.
2. Call `steer_and_generate(component, layer, alpha, prompt, head=...)`.
   - `alpha` is in units of the direction's own projection std (from
     training) — `alpha=1.0` is a moderate shift, `alpha=3-5` is strong,
     negative `alpha` pushes the opposite way.
   - `head` is only used for `component="mha"`.
3. Compare the returned `baseline` (unsteered) vs `steered` continuations
   for the same prompt. If steering is working, this comparison should show
   a directional difference (e.g. more hedging/agreement vs. more
   confidence) that scales with `alpha`.

**Tips:**
- Start with the layer/key that had the best probe accuracy
  (`probe_metadata.json`'s `*_best_key` fields, or the `metrics.json` written
  by `write_metrics`) — a direction a probe couldn't separate well is
  unlikely to steer anything meaningful either.
- If steering has no visible effect, try a larger `|alpha|` before
  concluding the direction doesn't matter — small shifts can be invisible in
  greedy-decoded text.
