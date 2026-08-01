# Skill: interpret_metrics

Every `train_sae` call and `read_sweep_log` entry reports three numbers:

- **fvu** (fraction of variance unexplained): reconstruction quality, lower
  is better. 0 = perfect reconstruction, 1 = no better than predicting the
  mean.
- **l0**: mean number of features active per sample. Should sit close to the
  `k` you requested; if it's meaningfully lower, many top-k slots are being
  filled with zero-activation features (undertrained, or `k` too large for
  this layer).
- **dead_frac**: fraction of the dictionary that never fires on the
  validation set. Healthy runs keep this well under 0.5; a high dead_frac
  wastes capacity and usually means more steps, or a larger `k_aux`/smaller
  `dead_steps`, are needed.

**Reasoning about a sweep:** there's no single best config — lower fvu at
the same k is better, and lower k at the same fvu is more interpretable/
sparse. Compare configs at a fixed k across dict_size, and at a fixed
dict_size across k, looking for the Pareto frontier (lowest fvu for each l0
level) rather than a single winner. Stop sweeping once new configs stop
improving that frontier, or once dead_frac stays low and fvu plateaus across
2-3 further configs.
