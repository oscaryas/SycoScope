"""
Train a TopK sparse autoencoder on one layer of SAE/pipeline/cache_activations.py's
frozen activation cache. Never touches the source model -- reads only the
cached memmaps + meta.json.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SAE.sae.data import DEFAULT_CACHE_DIR, ActivationSplit
from SAE.sae.model import TopKSAE

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "runs"

MANIFEST_KEYS = [
    # Identity of the run: architecture, data source, and optimization
    # hyperparameters. "steps" is deliberately excluded -- bumping it to
    # train an existing run longer is a supported resume, not a config change.
    "layer", "granularity", "dict_size", "k", "k_aux", "aux_coef", "dead_steps",
    "lr", "batch_size", "seed", "cache_dir",
]


def make_run_id(layer, granularity, dict_size, k, seed) -> str:
    key = f"layer{layer}_{granularity}_d{dict_size}_k{k}_seed{seed}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"{key}_{digest}"


@torch.no_grad()
def evaluate(model, split, batch_size, scale, mean, device) -> dict:
    model.eval()
    sq_err_sum = 0.0
    var_sum = 0.0
    l0_sum = 0.0
    n_rows = 0
    fired = torch.zeros(model.dict_size, dtype=torch.bool)
    mean_t = torch.from_numpy(mean).to(device)

    for batch in split.iter_batches("val", batch_size, shuffle=False):
        x = batch.to(device) * scale
        recon, code, _ = model(x)
        sq_err_sum += ((x - recon) ** 2).sum().item()
        var_sum += ((x - mean_t) ** 2).sum().item()
        l0_sum += (code != 0).float().sum().item()
        fired |= (code != 0).any(dim=0).cpu()
        n_rows += x.shape[0]

    model.train()
    return {
        "fvu": sq_err_sum / max(var_sum, 1e-8),
        "l0": l0_sum / max(n_rows, 1),
        "dead_frac": 1.0 - fired.float().mean().item(),
    }


def run_training(config: dict) -> dict:
    cfg = argparse.Namespace(**{"force": False, **config})

    cache_dir = Path(cfg.cache_dir)
    output_root = Path(cfg.output_root)
    run_id = make_run_id(cfg.layer, cfg.granularity, cfg.dict_size, cfg.k, cfg.seed)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {key: getattr(cfg, key) for key in MANIFEST_KEYS}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            if not cfg.force:
                raise RuntimeError(
                    f"Existing run at {run_dir} was built with a different config. "
                    "Re-run with force=True/--force to discard it and restart."
                )
            for p in run_dir.glob("*"):
                p.unlink()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split = ActivationSplit(cache_dir, cfg.layer, cfg.granularity, seed=cfg.seed)
    scale, mean = split.compute_stats(seed=cfg.seed)

    model = TopKSAE(split.hidden_size, cfg.dict_size, cfg.k).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    checkpoint_path = run_dir / "checkpoint.pt"
    step = 0
    last_fired = torch.zeros(cfg.dict_size, dtype=torch.long)
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        last_fired = ckpt["last_fired"]

    metrics_path = run_dir / "metrics.jsonl"
    warmup_steps = max(1, cfg.steps // 20)

    while step < cfg.steps:
        for batch in split.iter_batches("train", cfg.batch_size, shuffle=True, seed=cfg.seed + step):
            if step >= cfg.steps:
                break

            x = batch.to(device) * scale
            lr_scale = min(1.0, (step + 1) / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * lr_scale

            recon, code, pre_acts = model(x)
            recon_loss = F.mse_loss(recon, x)
            dead_mask = (last_fired > cfg.dead_steps).to(device)
            aux = model.aux_loss(x, recon, pre_acts, dead_mask, cfg.k_aux)
            loss = recon_loss + cfg.aux_coef * aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.renormalize_decoder_()

            fired_now = (code != 0).any(dim=0).cpu()
            last_fired += 1
            last_fired[fired_now] = 0

            step += 1

            if step % cfg.eval_every == 0 or step == cfg.steps:
                eval_metrics = evaluate(model, split, cfg.batch_size, scale, mean, device)
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "loss": loss.item(), **eval_metrics}) + "\n")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "last_fired": last_fired,
                    },
                    checkpoint_path,
                )

    final_metrics = evaluate(model, split, cfg.batch_size, scale, mean, device)
    final_metrics.update({"run_id": run_id, "step": step})
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    sweep_log_path = output_root / "sweep_log.jsonl"
    with open(sweep_log_path, "a", encoding="utf-8") as f:
        entry = {**manifest, **final_metrics, "created_at": datetime.now(timezone.utc).isoformat()}
        f.write(json.dumps(entry) + "\n")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--granularity", choices=["sentence", "response"], required=True)
    parser.add_argument("--dict-size", type=int, default=16384)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--k-aux", type=int, default=256)
    parser.add_argument("--aux-coef", type=float, default=1.0 / 32)
    parser.add_argument(
        "--dead-steps", type=int, default=200,
        help="steps without winning top-k before a feature counts as dead for the aux loss",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    metrics = run_training(vars(args))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
