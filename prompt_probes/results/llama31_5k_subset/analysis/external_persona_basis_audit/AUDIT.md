# External Lu-style persona basis audit

**Status: numerically valid, but extraction-layer provenance is unresolved. No projected variance result has been computed.**

## Confirmed

- Public artifact targets `meta-llama/Llama-3.1-8B-Instruct` according to its model configuration.
- Pinned dataset revision: `98ae1b599e8ed1ae6952d83167ea4e4ba42111ce`.
- PCA artifact SHA256: `327d15905d1840ff4c1501e8522d3b9948282fc69f8aeb02d2b1b7a5fe5eb09f`.
- Three finite orthonormal 4096-dimensional PCs; role-mean vector also has 4096 dimensions.
- 275 source roles. Variance fractions: PC1 27.58%, PC2 16.83%, PC3 7.04%; total 51.45% **of source role-vector variance**, not of our response variance.
- Artifact metadata and PCA construction source specify mean-over-assistant-token extraction. This is the appropriate pooling family for our response-mean analysis.
- PC1 cosine with the source default-minus-role axis is 0.818; PC1 is not identical to that axis.
- Loaded using PyTorch's restricted weights-only loader with standard NumPy array constructors explicitly allowed. External source files were read, not executed. A numeric-only NPZ is retained for future analysis.

## Why the analysis is paused

The artifact does not record the layer at which its source role vectors were extracted. The new steering runner's configuration says layer 16, but the PCA builder reads vectors from a separate prior project without selecting or checking a layer. The available prior-project configuration does not include this Llama-3.1-8B run. Thus we cannot distinguish `hidden_states[16]` (our block 15, not cached) from block 16 output (our `response_L16`), or independently establish the original extraction depth. Dimensionality alone cannot settle this.

The surrounding project documents an off-by-one steering-layer error. Its public alpha=0 role-vector cloud also uses last-token extraction, not the response-mean extraction used for this basis. Neither resolves the source-layer ambiguity.

The prior pipeline explicitly omits the role-expression judge/filter. This is an independent **Lu-style adaptation**, not a verified exact replication of Lu et al. The original model commit, actual Llama-8B extraction configuration, and role-filtering logs are not established by the audited materials.

## Required next step

Obtain the author's extraction command/log specifying model revision, zero-based block index (or `hidden_states` index), response-token span and role filtering; ideally obtain the original response-mean role vectors. Alternatively, construct a new independently elicited Lu-style basis on our model with documented extraction settings. Do not choose a layer by whichever produces the strongest system-prompt variance.

After matching the layer, use the frozen 400-question × 40-system-condition cohort. Project responses and repeat the system/question/interaction decomposition, reporting both variance shares inside the three-PC subspace and the fraction of full activation variance captured. Include random three-dimensional subspaces and question-bootstrap uncertainty. Do not reuse a single middle-layer basis as if it were a validated basis for all layers.

## Sources

- [Pinned dataset card](https://huggingface.co/datasets/pandaman007/steered-persona-space/blob/98ae1b599e8ed1ae6952d83167ea4e4ba42111ce/README.md)
- [PCA construction](https://github.com/pauljCherian/steered-persona-space/blob/c8f975e88b4b4c366befaa59342c4405d7705def/scripts/compute_baseline_pca.py)
- [Baseline loader](https://github.com/pauljCherian/steered-persona-space/blob/c8f975e88b4b4c366befaa59342c4405d7705def/_common.py)
- [Prior extraction pipeline](https://github.com/pauljCherian/persona-space-comparison/blob/b94fd93ff532cd51edf93c7b7a3e10c1b526d086/pipeline.sh)

The original activation-variance results and the running GPU experiment were not changed.
