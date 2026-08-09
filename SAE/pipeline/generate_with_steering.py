"""
Batch-generate steered responses across datasets for a config of datasets x
steering vectors (trained SAE latents), and save them to
SAE/results/steered_responses/<run+latent_id>/<dataset>.jsonl.

Each vector's coeff = coeff_multiplier * p90, where p90 is read from the
trained run's metrics.json (written by compute_steering_coeffs.py) rather
than recomputed here -- run that script first for any run referenced in
--config. coeff_multiplier defaults to steer.py's DEFAULT_COEFF_MULTIPLIER
and can be overridden per vector, so multipliers can be swept at generation
time without touching the activations cache. A vector may instead set an
explicit "coeff" to bypass p90 entirely.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from steer import DEFAULT_COEFF_MULTIPLIER, generate_with_steering as steer_generate

from utils import datasets, inference
from utils.model import DEFAULT_MODEL, cleanup, load_model_and_tokenizer
from utils.sae_utils import DEFAULT_ACTIVATIONS_DIR, DEFAULT_TRAINED_SAE_DIR, sae_dir_name

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "steered_responses"
DEFAULT_LIMIT = 300


def _run_tag(vector: dict) -> str:
    return sae_dir_name(
        vector["layer"], vector["n_latents"], vector.get("k", 3), vector.get("centered", False), vector.get("seed", 0)
    )


def run_name_for(vector: dict) -> str:
    return f"{_run_tag(vector)}_latent{vector['latent_id']}"


def resolve_coeff(vector: dict, trained_sae_dir: Path) -> dict:
    """
    Returns {"coeff", "coeff_source", "p90", "coeff_multiplier"}. "p90" and
    "coeff_multiplier" are None when "coeff" was set explicitly (no p90
    lookup happens in that case).
    """
    if "coeff" in vector:
        return {"coeff": float(vector["coeff"]), "coeff_source": "explicit", "p90": None, "coeff_multiplier": None}

    run_dir = trained_sae_dir / _run_tag(vector)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics.json at {run_dir} -- train this SAE run first")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "steering_p90" not in metrics:
        raise RuntimeError(
            f"{metrics_path} has no 'steering_p90' field -- run "
            f"`python SAE/pipeline/compute_steering_coeffs.py --run-dir {run_dir}` first "
            '(or pass an explicit "coeff" for this vector in the config).'
        )
    latent_id = vector["latent_id"]
    p90s = metrics["steering_p90"]
    if latent_id >= len(p90s):
        raise IndexError(f"latent_id {latent_id} out of range for {metrics_path} ({len(p90s)} latents)")

    p90 = float(p90s[latent_id])
    coeff_multiplier = float(vector.get("coeff_multiplier", DEFAULT_COEFF_MULTIPLIER))
    return {
        "coeff": coeff_multiplier * p90,
        "coeff_source": "p90_multiplier",
        "p90": p90,
        "coeff_multiplier": coeff_multiplier,
    }


def load_latent_label(vector: dict, trained_sae_dir: Path) -> tuple[str | None, str | None]:
    labels_path = trained_sae_dir / _run_tag(vector) / "labels.json"
    if not labels_path.exists():
        return None, None
    for rec in json.loads(labels_path.read_text(encoding="utf-8")):
        if rec["latent_id"] == vector["latent_id"]:
            return rec.get("title"), rec.get("description")
    return None, None


def generate_for_vector(
    vector: dict,
    dataset_files: list[str],
    model,
    tokenizer,
    output_dir: Path,
    trained_sae_dir: Path,
    activations_dir: Path,
    limit: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model_name: str,
):
    layer = vector["layer"]
    n_latents = vector["n_latents"]
    latent_id = vector["latent_id"]
    k = vector.get("k", 3)
    centered = vector.get("centered", False)
    seed = vector.get("seed", 0)

    resolved = resolve_coeff(vector, trained_sae_dir)
    coeff = resolved["coeff"]
    title, description = load_latent_label(vector, trained_sae_dir)

    run_name = run_name_for(vector)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    steering_config = {
        "layer": layer, "n_latents": n_latents, "k": k, "centered": centered, "seed": seed,
        "latent_id": latent_id, **resolved,
        "title": title, "description": description,
        "model": model_name, "limit": limit,
        "datasets": [Path(f).stem for f in dataset_files],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "steering_config.json").write_text(json.dumps(steering_config, indent=2), encoding="utf-8")

    for filename in dataset_files:
        dataset_name = Path(filename).stem
        records = list(datasets.iter_prompts(filename))[:limit]
        out_path = run_dir / f"{dataset_name}.jsonl"
        print(f"  [{run_name}] {dataset_name}: generating {len(records)} responses (coeff={coeff:.3f}, {resolved['coeff_source']})")
        with open(out_path, "w", encoding="utf-8") as f:
            for batch in inference.iter_batches(records, batch_size):
                responses = steer_generate(
                    [r["text"] for r in batch],
                    layer=layer, n_latents=n_latents, latent_id=latent_id, coeff=coeff,
                    k=k, centered=centered, seed=seed,
                    model=model, tokenizer=tokenizer, model_name=model_name,
                    trained_sae_dir=trained_sae_dir, activations_dir=activations_dir,
                    max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                    verbose=False,
                )
                for record, response in zip(batch, responses):
                    f.write(
                        json.dumps(
                            {
                                "dataset": record["dataset"],
                                "row_id": record["row_id"],
                                "prompt_col": record["prompt_col"],
                                "prompt": record["text"],
                                "sample_idx": 0,
                                "response": response,
                                "model": model_name,
                            }
                        )
                        + "\n"
                    )
        print(f"  Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help='JSON config: {"datasets": [...], "vectors": [...]}')
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--trained-sae-dir", default=str(DEFAULT_TRAINED_SAE_DIR))
    parser.add_argument("--activations-dir", default=str(DEFAULT_ACTIVATIONS_DIR))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Prompts per dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    default_dataset_files = config.get("datasets") or datasets.list_dataset_files()
    vectors = config["vectors"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_sae_dir = Path(args.trained_sae_dir)
    activations_dir = Path(args.activations_dir)

    print(f"Loading model {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    for vector in vectors:
        dataset_files = vector.get("datasets") or default_dataset_files
        generate_for_vector(
            vector, dataset_files, model, tokenizer, output_dir, trained_sae_dir, activations_dir,
            limit=args.limit, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, model_name=args.model,
        )

    cleanup(model, tokenizer)
    print(f"Done: {len(vectors)} vector(s) processed.")


if __name__ == "__main__":
    main()
