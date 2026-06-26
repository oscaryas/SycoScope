# Skill: train_residual_probes

Call `train_probe_family("residual")`. Reads `activations/residual.npy` from cache.

**What it trains:** One linear probe per layer from the residual stream.

**Training config:** Same as MHA/MLP.

**Activation source:** `output_hidden_states=True`, `hidden_states[layer+1]` at answer position.
Shape: (n_layers, n_examples, hidden_dim)
