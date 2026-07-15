"""
Train a single Top-K SAE against a frozen SAE/results/activations/ artifact
(written by cache_activations.py). CPU/small-GPU work only -- never touches
the LLM. Writes trained weights + metadata to SAE/results/trained_sae/.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sae_utils import (
    DEFAULT_ACTIVATIONS_DIR,
    DEFAULT_TRAINED_SAE_DIR,
    TopKSAE,
    compute_global_center,
    load_layer_array,
    sae_dir_name,
)

# Fixed, independent of the run's own `seed`, so different-seed runs of the
# same config are compared against an identical held-out set.
SPLIT_SEED = 0
EVAL_CHUNK = 4096


def train_sae(
    layer: int,
    n_latents: int,
    centered: bool,
    seed: int,
    k: int = 3,
    activations_dir: Path = DEFAULT_ACTIVATIONS_DIR,
    output_dir: Path = DEFAULT_TRAINED_SAE_DIR,
    batch_size: int = 512,
    max_epochs: int = 300,
    patience: int = 10,
    val_frac: float = 0.05,
    device: str | None = None,
    force: bool = False,
) -> Path:
    run_dir = output_dir / sae_dir_name(layer, n_latents, k, centered, seed)
    if not force and (run_dir / "metrics.json").exists():
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    arr = load_layer_array(layer, activations_dir)
    d_in = arr.shape[1]
    center = compute_global_center(layer, activations_dir) if centered else np.zeros(d_in, dtype=np.float32)
    x = torch.from_numpy(arr.astype(np.float32)) - torch.from_numpy(center)
    n = x.shape[0]

    split_gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n, generator=split_gen)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    x_train, x_val = x[train_idx], x[val_idx]

    torch.manual_seed(seed)
    model = TopKSAE(d_in, n_latents, k).to(device)
    lr = 2e-4 / (n_latents / 2**14) ** 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_val_dev = x_val.to(device)
    train_loader = DataLoader(
        TensorDataset(x_train), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    fired_ever = torch.zeros(n_latents, dtype=torch.bool, device=device)
    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0
    n_epochs_trained = 0
    early_stopped = False

    for epoch in range(max_epochs):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            xhat, z = model(xb)
            loss = F.mse_loss(xhat, xb)
            loss.backward()
            optimizer.step()
            model.normalize_decoder_()
            fired_ever |= (z != 0).any(dim=0)

        model.eval()
        with torch.no_grad():
            xhat_val, _ = model(x_val_dev)
            val_loss = F.mse_loss(xhat_val, x_val_dev).item()

        n_epochs_trained = epoch + 1
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = {name: t.detach().clone() for name, t in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                early_stopped = True
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        z_chunks, xhat_chunks = [], []
        for i in range(0, n, EVAL_CHUNK):
            xb = x[i : i + EVAL_CHUNK].to(device)
            xhat_b, z_b = model(xb)
            z_chunks.append(z_b.cpu())
            xhat_chunks.append(xhat_b.cpu())
        z_full = torch.cat(z_chunks)
        xhat_full = torch.cat(xhat_chunks)

    recon_err = ((xhat_full - x) ** 2).sum(dim=1)
    total_var = ((x - x.mean(dim=0)) ** 2).sum(dim=1)
    explained_variance = 1 - (recon_err.sum() / total_var.sum()).item()

    firing_rate = (z_full != 0).float().mean(dim=0).numpy()
    dead_latent_count = int((~fired_ever.cpu()).sum().item())
    mean_l0 = float((z_full != 0).sum(dim=1).float().mean().item())

    top1_val, top1_idx = z_full.max(dim=1)
    assignments = torch.where(top1_val > 0, top1_idx, torch.full_like(top1_idx, -1)).numpy().astype(np.int32)
    assignment_strength = torch.where(top1_val > 0, top1_val, torch.zeros_like(top1_val)).numpy().astype(np.float32)

    topk_vals, topk_idx = z_full.topk(k, dim=1)
    assignments_topk = torch.where(topk_vals > 0, topk_idx, torch.full_like(topk_idx, -1)).numpy().astype(np.int32)

    np.save(run_dir / "assignments.npy", assignments)
    np.save(run_dir / "assignment_strength.npy", assignment_strength)
    np.save(run_dir / "assignments_topk.npy", assignments_topk)
    torch.save(model.state_dict(), run_dir / "sae.pt")

    meta_path = activations_dir / "meta.json"
    activations_input_hash = json.loads(meta_path.read_text(encoding="utf-8"))["input_hash"] if meta_path.exists() else None

    config = {
        "layer": layer, "n_latents": n_latents, "k": k, "centered": centered, "seed": seed,
        "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
        "val_frac": val_frac, "d_in": d_in, "n_epochs_trained": n_epochs_trained,
        "activations_input_hash": activations_input_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics = {
        "final_recon_loss": best_val_loss,
        "explained_variance": explained_variance,
        "firing_rate": firing_rate.tolist(),
        "dead_latent_count": dead_latent_count,
        "mean_l0": mean_l0,
        "n_epochs_trained": n_epochs_trained,
        "early_stopped": early_stopped,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-latents", type=int, required=True)
    parser.add_argument("--centered", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--activations-dir", default=str(DEFAULT_ACTIVATIONS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_TRAINED_SAE_DIR))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = train_sae(
        layer=args.layer, n_latents=args.n_latents, centered=args.centered, seed=args.seed, k=args.k,
        activations_dir=Path(args.activations_dir), output_dir=Path(args.output_dir),
        batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience,
        val_frac=args.val_frac, force=args.force,
    )
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
