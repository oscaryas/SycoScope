"""
Synthetic-data smoke test for the SAE training loop. Builds a tiny fake
activation cache on disk (same file layout cache_activations.py produces) so
the full data -> model -> train path is exercised without a GPU, the real
8B model, or the real activation cache.
"""

import json

import numpy as np
import pandas as pd
import torch

from SAE.sae.model import TopKSAE
from SAE.sae.train import run_training

HIDDEN_SIZE = 16
DICT_SIZE = 32
N_RESPONSES = 40
SENTS_PER_RESPONSE = 3


def _build_synthetic_cache(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_sentences = N_RESPONSES * SENTS_PER_RESPONSE

    rows = [
        {"response_id": f"resp_{r}", "global_idx": r * SENTS_PER_RESPONSE + s}
        for r in range(N_RESPONSES)
        for s in range(SENTS_PER_RESPONSE)
    ]
    pd.DataFrame(rows).to_parquet(cache_dir / "sentences.parquet", index=False)

    rng = np.random.default_rng(0)
    # A few latent directions plus small noise, so reconstruction loss has
    # real structure to learn instead of being pure noise.
    latents = rng.normal(size=(n_sentences, 4)).astype(np.float32)
    directions = rng.normal(size=(4, HIDDEN_SIZE)).astype(np.float32)
    activations = latents @ directions + 0.01 * rng.normal(size=(n_sentences, HIDDEN_SIZE)).astype(np.float32)

    arr = np.lib.format.open_memmap(
        cache_dir / "layer_00.npy", mode="w+", dtype=np.float16, shape=(n_sentences, HIDDEN_SIZE)
    )
    arr[:] = activations.astype(np.float16)
    arr.flush()

    meta = {"layers": [0], "hidden_size": HIDDEN_SIZE}
    (cache_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _base_config(cache_dir, output_root):
    return {
        "layer": 0,
        "granularity": "sentence",
        "dict_size": DICT_SIZE,
        "k": 4,
        "k_aux": 8,
        "aux_coef": 1.0 / 32,
        "dead_steps": 5,
        "lr": 1e-2,
        "batch_size": 16,
        "steps": 30,
        "eval_every": 10,
        "cache_dir": str(cache_dir),
        "output_root": str(output_root),
        "seed": 0,
    }


def test_loss_decreases_and_decoder_stays_unit_norm(tmp_path):
    cache_dir = tmp_path / "activations"
    _build_synthetic_cache(cache_dir)
    output_root = tmp_path / "runs"

    metrics = run_training(_base_config(cache_dir, output_root))

    run_dirs = list(output_root.glob("layer0_sentence_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    first_fvu = json.loads(lines[0])["fvu"]
    last_fvu = json.loads(lines[-1])["fvu"]
    assert last_fvu < first_fvu, f"FVU should improve with training: {first_fvu} -> {last_fvu}"

    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    model = TopKSAE(HIDDEN_SIZE, DICT_SIZE, 4)
    model.load_state_dict(ckpt["model"])
    norms = model.decoder.weight.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    assert (output_root / "sweep_log.jsonl").exists()
    assert metrics["step"] == 30


def test_resume_continues_step_count_instead_of_restarting(tmp_path):
    cache_dir = tmp_path / "activations"
    _build_synthetic_cache(cache_dir)
    output_root = tmp_path / "runs"

    config = _base_config(cache_dir, output_root)
    config["steps"] = 10
    run_training(config)

    config["steps"] = 20
    metrics = run_training(config)
    assert metrics["step"] == 20

    run_dirs = list(output_root.glob("layer0_sentence_*"))
    assert len(run_dirs) == 1
    lines = (run_dirs[0] / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    steps_logged = [json.loads(line)["step"] for line in lines]
    assert steps_logged == sorted(steps_logged)
    assert steps_logged[0] < 20
    assert steps_logged[-1] == 20


def test_mismatched_config_requires_force(tmp_path):
    cache_dir = tmp_path / "activations"
    _build_synthetic_cache(cache_dir)
    output_root = tmp_path / "runs"

    config = _base_config(cache_dir, output_root)
    run_training(config)

    config["lr"] = 5e-2
    try:
        run_training(config)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "changing a training hyperparameter without force=True should raise"

    config["force"] = True
    metrics = run_training(config)
    assert metrics["step"] == config["steps"]
