"""
Precompute each latent's p90 firing strength (via steer.py's
compute_latent_activation_stats) for a trained SAE run, and persist it into
that run's metrics.json -- so generate_with_steering.py can read a fixed
per-latent p90 and cheaply compute coeff = multiplier * p90 at steering time,
instead of recomputing p90 (a torch.load + SAE forward pass over the firing
members) on every generation batch. Keeping the multiplier out of this step
means it can be swept at generation time without rerunning this script (and
without needing the activations cache again).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from label_clusters import expand_layer_runs
from steer import compute_latent_activation_stats
from utils.sae_utils import DEFAULT_ACTIVATIONS_DIR, DEFAULT_TRAINED_SAE_DIR


def compute_run_p90s(
    run_dir: Path,
    activations_dir: Path = DEFAULT_ACTIVATIONS_DIR,
    trained_sae_dir: Path = DEFAULT_TRAINED_SAE_DIR,
    force: bool = False,
) -> Path:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    layer, n_latents, k, centered, seed = (
        config["layer"], config["n_latents"], config["k"], config["centered"], config["seed"],
    )

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "steering_p90" in metrics and not force:
        print(f"{run_dir.name}: steering_p90 already present, skipping (use --force to recompute)")
        return metrics_path

    p90s = []
    for latent_id in range(n_latents):
        stats = compute_latent_activation_stats(
            layer, n_latents, latent_id, k, centered, seed, trained_sae_dir, activations_dir
        )
        p90s.append(stats["p90"])

    metrics["steering_p90"] = p90s
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # p90 == 0 means the latent never fires, or is so marginal it drops out of the top-k on
    # re-encode. Either way it has no natural scale to calibrate against, so generation must
    # pass an explicit coeff to steer with it.
    n_zero = sum(1 for p in p90s if p == 0.0)
    suffix = f" ({n_zero} with p90=0, not steerable by multiplier)" if n_zero else ""
    print(f"{run_dir.name}: wrote steering_p90 for {n_latents} latents{suffix}")
    return metrics_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--trained-sae-dir", default=str(DEFAULT_TRAINED_SAE_DIR))
    parser.add_argument("--activations-dir", default=str(DEFAULT_ACTIVATIONS_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if (args.run_dir is None) == (args.layer is None):
        parser.error("specify exactly one of --run-dir or --layer")

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        run_dirs = expand_layer_runs(args.layer, Path(args.trained_sae_dir))
        if not run_dirs:
            parser.error(f"no run directories found for layer {args.layer} under {args.trained_sae_dir}")

    for run_dir in run_dirs:
        compute_run_p90s(
            run_dir,
            activations_dir=Path(args.activations_dir),
            trained_sae_dir=Path(args.trained_sae_dir),
            force=args.force,
        )
    print(f"Done: processed {len(run_dirs)} run(s).")


if __name__ == "__main__":
    main()
