# Skill: train_mlp_probes

Call `train_probe_family("mlp")`. Reads `activations/mlp.npy` from cache.

**What it trains:** One linear probe per layer.

**Training config:** Same as MHA — BCEWithLogitsLoss, Adam lr=0.001, 25 epochs, batch 25, 80/20 split.

**Activation source:** Output of `mlp.down_proj` at answer token position.
Shape: (n_layers, n_examples, hidden_dim)
