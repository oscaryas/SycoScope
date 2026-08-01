"""
Generate text with an SAE latent's decoder direction added to the residual
stream, to see how activating that feature changes generation.

Adds `coeff * decoder_direction` to every token position at the output of
decoder block `layer - 1` (i.e. hidden_states[layer] in the HF convention
used everywhere else in SAE/pipeline, where hidden_states[0] = embeddings).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import inference
from utils.model import DEFAULT_MODEL, cleanup, load_model_and_tokenizer
from utils.sae_utils import (
    DEFAULT_ACTIVATIONS_DIR,
    DEFAULT_TRAINED_SAE_DIR,
    TopKSAE,
    compute_global_center,
    sae_dir_name,
)

DEFAULT_COEFF_MULTIPLIER = 4.0


def load_decoder_direction(
    layer: int,
    n_latents: int,
    latent_id: int,
    k: int = 3,
    centered: bool = False,
    seed: int = 0,
    trained_sae_dir: Path = DEFAULT_TRAINED_SAE_DIR,
) -> torch.Tensor:
    run_dir = trained_sae_dir / sae_dir_name(layer, n_latents, k, centered, seed)
    state_dict = torch.load(run_dir / "sae.pt", map_location="cpu")
    w_dec = state_dict["W_dec"]  # (d_in, n_latents), columns already unit-norm
    return w_dec[:, latent_id]


def compute_latent_activation_stats(
    layer: int,
    n_latents: int,
    latent_id: int,
    k: int = 3,
    centered: bool = False,
    seed: int = 0,
    trained_sae_dir: Path = DEFAULT_TRAINED_SAE_DIR,
    activations_dir: Path = DEFAULT_ACTIVATIONS_DIR,
    device: str | None = None,
) -> dict:
    """
    Real firing-strength stats for one latent, computed only over the samples
    where it actually fires (per assignments_topk.npy), by re-encoding just
    those rows through the trained SAE. Used to calibrate `coeff` in
    generate_with_steering to the scale this latent actually produces, rather
    than guessing an arbitrary constant.
    """
    run_dir = trained_sae_dir / sae_dir_name(layer, n_latents, k, centered, seed)
    assignments_topk = np.load(run_dir / "assignments_topk.npy")
    n_total = assignments_topk.shape[0]
    member_idx = np.flatnonzero((assignments_topk == latent_id).any(axis=1))

    if member_idx.size == 0:
        return {"firing_rate": 0.0, "n_fired": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}

    state_dict = torch.load(run_dir / "sae.pt", map_location="cpu")
    d_in = state_dict["b_dec"].shape[0]
    model = TopKSAE(d_in, n_latents, k)
    model.load_state_dict(state_dict)
    model.eval()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # mmap so we only pull the (small) subset of rows where this latent fires,
    # instead of loading the full (n_sentences, d_in) activations array.
    arr = np.load(activations_dir / f"layer_{layer:02d}.npy", mmap_mode="r")
    x_members = torch.from_numpy(np.asarray(arr[member_idx], dtype=np.float32))
    if centered:
        x_members -= torch.from_numpy(compute_global_center(layer, activations_dir))

    with torch.no_grad():
        z, _ = model.encode(x_members.to(device))
    z_lat = z[:, latent_id].cpu()
    z_lat = z_lat[z_lat > 0]  # guards float edge cases; should be ~all of member_idx by construction

    return {
        "firing_rate": member_idx.size / n_total,
        "n_fired": int(z_lat.numel()),
        "mean": z_lat.mean().item(),
        "median": z_lat.median().item(),
        "p90": torch.quantile(z_lat, 0.9).item(),
        "max": z_lat.max().item(),
    }


class _AddDirectionHook:
    """Forward hook that adds `coeff * direction` to a decoder layer's output."""

    def __init__(self, direction: torch.Tensor, coeff: float):
        self.direction = direction
        self.coeff = coeff

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden + self.coeff * self.direction.to(dtype=hidden.dtype, device=hidden.device)
        return (hidden, *output[1:]) if is_tuple else hidden


def generate_with_steering(
    prompts: list[str],
    layer: int,
    n_latents: int,
    latent_id: int,
    coeff: float | None = None,
    coeff_multiplier: float | None = DEFAULT_COEFF_MULTIPLIER,
    k: int = 3,
    centered: bool = False,
    seed: int = 0,
    model=None,
    tokenizer=None,
    model_name: str = DEFAULT_MODEL,
    trained_sae_dir: Path = DEFAULT_TRAINED_SAE_DIR,
    activations_dir: Path = DEFAULT_ACTIVATIONS_DIR,
    device: str | None = None,
    verbose: bool = True,
    **generate_kwargs,
) -> list[str]:
    """
    Generate completions for `prompts` with an SAE latent's decoder direction
    steered into the residual stream. Pass an already-loaded `model`/
    `tokenizer` (e.g. from a notebook) to avoid reloading the 8B model per call.

    Coefficient: pass `coeff` for a raw scaling constant. Otherwise (the
    default) `coeff` is derived automatically as
    `coeff_multiplier * p90(this latent's real firing strengths)`, via
    compute_latent_activation_stats -- this keeps the steering magnitude on
    the scale the latent actually produces instead of a guessed constant.
    """
    if coeff is None:
        stats = compute_latent_activation_stats(
            layer, n_latents, latent_id, k, centered, seed, trained_sae_dir, activations_dir, device
        )
        if stats["n_fired"] == 0:
            raise ValueError(
                f"latent {latent_id} never fires in the cached activations (per assignments_topk.npy) "
                "-- pass an explicit `coeff` if you want to steer with it anyway."
            )
        coeff = coeff_multiplier * stats["p90"]
        if verbose:
            print(
                f"[steer] latent {latent_id} fires on {stats['firing_rate']:.2%} of samples "
                f"(p90={stats['p90']:.3f}, max={stats['max']:.3f}) -> coeff={coeff:.3f} "
                f"({coeff_multiplier}x p90)"
            )

    own_model = model is None
    if own_model:
        model, tokenizer = load_model_and_tokenizer(model_name)

    direction = load_decoder_direction(layer, n_latents, latent_id, k, centered, seed, trained_sae_dir)
    target_layer = model.model.layers[layer - 1]
    hook = _AddDirectionHook(direction, coeff)
    handle = target_layer.register_forward_hook(hook)
    try:
        return inference.generate_batch(model, tokenizer, prompts, **generate_kwargs)
    finally:
        handle.remove()
        if own_model:
            cleanup(model, tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-latents", type=int, required=True)
    parser.add_argument("--latent-id", type=int, required=True)
    coeff_group = parser.add_mutually_exclusive_group()
    coeff_group.add_argument("--coeff", type=float, default=None, help="Raw scaling coefficient for the decoder direction")
    coeff_group.add_argument(
        "--coeff-multiplier",
        type=float,
        default=DEFAULT_COEFF_MULTIPLIER,
        help="If --coeff isn't given, coeff = this * p90(latent's real firing strengths)",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--centered", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", action="append", required=True, dest="prompts")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trained-sae-dir", default=str(DEFAULT_TRAINED_SAE_DIR))
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    baseline = inference.generate_batch(model, tokenizer, args.prompts, max_new_tokens=args.max_new_tokens)
    steered = generate_with_steering(
        args.prompts,
        layer=args.layer,
        n_latents=args.n_latents,
        latent_id=args.latent_id,
        coeff=args.coeff,
        coeff_multiplier=args.coeff_multiplier,
        k=args.k,
        centered=args.centered,
        seed=args.seed,
        model=model,
        tokenizer=tokenizer,
        trained_sae_dir=Path(args.trained_sae_dir),
        max_new_tokens=args.max_new_tokens,
    )

    for prompt, base, steer in zip(args.prompts, baseline, steered):
        print(f"\n=== prompt: {prompt[:80]!r} ===")
        print(f"--- baseline ---\n{base}")
        print(f"--- steered (layer={args.layer} n_latents={args.n_latents} latent={args.latent_id}) ---\n{steer}")

    cleanup(model, tokenizer)


if __name__ == "__main__":
    main()
