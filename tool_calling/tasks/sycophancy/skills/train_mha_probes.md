# Skill: train_mha_probes

Call `train_probe_family("mha")`. Reads `activations/mha.npy` from cache.

**What it trains:** One linear probe per (layer, head). For 48 layers × 16 heads = 768 probes.

**Training config (paper exact):**
- Loss: BCEWithLogitsLoss, Optimizer: Adam lr=0.001
- Epochs: 25, Batch size: 25, Split: 80/20

**Activation source:** Input to `self_attn.o_proj` at the answer token position.
Shape: (n_layers, n_heads, n_examples, head_dim)
