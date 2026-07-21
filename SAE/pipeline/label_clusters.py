import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llm_judge import DEFAULT_JUDGE_MODEL, call_judge, parse_json_response
from utils.sae_utils import DEFAULT_ACTIVATIONS_DIR, DEFAULT_TRAINED_SAE_DIR

JUDGE_SYSTEM_PROMPT = (
    "You are an interpretability researcher analyzing latent directions in a sparse "
    "autoencoder trained on a language model's hidden activations."
)


def gather_exemplars(
    assignments: np.ndarray,
    assignment_strength: np.ndarray,
    sentences_df: pd.DataFrame,
    latent_id: int,
    n_top: int = 100,
    n_random: int = 100,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    member_idx = np.flatnonzero(assignments == latent_id)
    if member_idx.size == 0:
        return [], []

    order = np.argsort(-assignment_strength[member_idx], kind="stable")
    top_idx = member_idx[order[:n_top]]

    remaining_idx = member_idx[~np.isin(member_idx, top_idx)]
    n_random_take = min(n_random, remaining_idx.size)
    if n_random_take > 0:
        rng = np.random.default_rng((seed, latent_id))
        random_idx = rng.choice(remaining_idx, size=n_random_take, replace=False)
    else:
        random_idx = np.array([], dtype=int)

    text_arr = sentences_df["text"].to_numpy()
    return text_arr[top_idx].tolist(), text_arr[random_idx].tolist()


def build_prompt1(top_sentences: list[str], random_sentences: list[str]) -> str:
    numbered = []
    for s in top_sentences:
        numbered.append(f"[top-activating] {s}")
    for s in random_sentences:
        numbered.append(f"[random from cluster] {s}")
    sentence_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(numbered))

    if random_sentences:
        provenance = (
            f"{len(top_sentences)} are the highest-activating examples for this latent; "
            f"{len(random_sentences)} more are a random sample from the rest of the cluster."
        )
    else:
        provenance = (
            f"These are all {len(top_sentences)} sentences currently assigned to this latent "
            "(the cluster is smaller than the requested sample size, so no random sample was needed)."
        )

    return f"""Below are sentences that activate a specific latent direction in a sparse \
autoencoder trained on a language model's hidden activations. {provenance}

Your task is to identify the precise function these sentences serve -- the shared reasoning \
strategy, linguistic pattern, or functional role -- and NOT the surface-level topic they happen \
to discuss. Sentences in this cluster may span many different subjects; look for what they are \
doing (e.g. "proposing an alternative approach", "expressing uncertainty", "restating the \
question before answering it"), not what they are about.

Sentences:
{sentence_block}

Respond with a single JSON object and nothing else, using exactly these two keys:
- "title": a crisp, single-concept title for this function (a few words; no slashes, no \
parentheses, no compound "X / Y" phrases -- pick the single best description).
- "description": 3-4 sentences that (1) state the specific function this latent captures, \
(2) describe what kinds of sentences ARE included in this cluster, and (3) describe what is \
explicitly NOT included -- what distinguishes this cluster from superficially similar sentences.

Return only the JSON object, with no markdown code fences and no additional commentary."""


def label_latent(
    latent_id: int,
    assignments: np.ndarray,
    assignment_strength: np.ndarray,
    sentences_df: pd.DataFrame,
    n_top: int,
    n_random: int,
    seed: int,
    judge_model: str,
) -> dict:
    cluster_size = int((assignments == latent_id).sum())
    top_sentences, random_sentences = gather_exemplars(
        assignments, assignment_strength, sentences_df, latent_id, n_top, n_random, seed
    )
    n_exemplars_used = len(top_sentences) + len(random_sentences)

    record = {
        "latent_id": latent_id,
        "cluster_size": cluster_size,
        "n_exemplars_used": n_exemplars_used,
        "judge_model": judge_model,
        "title": None,
        "description": None,
        "status": "ok",
        "error": None,
    }

    if n_exemplars_used == 0:
        record["status"] = "skipped_empty_cluster"
        return record

    try:
        raw = call_judge(build_prompt1(top_sentences, random_sentences), system=JUDGE_SYSTEM_PROMPT, model=judge_model)
        parsed = parse_json_response(raw)
        title = parsed.get("title")
        description = parsed.get("description")
        if not title or not isinstance(title, str) or not description or not isinstance(description, str):
            raise ValueError(f"judge response missing non-empty title/description: {parsed!r}")
        record["title"] = title
        record["description"] = description
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"

    return record


def label_run(
    run_dir: Path,
    sentences_df: pd.DataFrame,
    n_top: int = 100,
    n_random: int = 100,
    seed: int = 0,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    force: bool = False,
) -> Path:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    n_latents = config["n_latents"]
    assignments = np.load(run_dir / "assignments.npy")
    assignment_strength = np.load(run_dir / "assignment_strength.npy")
    if len(assignments) != len(sentences_df) or len(assignment_strength) != len(sentences_df):
        raise ValueError(
            f"{run_dir}: assignments length {len(assignments)} != sentences.parquet length {len(sentences_df)}"
        )

    labels_path = run_dir / "labels.json"
    existing: dict[int, dict] = {}
    if labels_path.exists() and not force:
        for rec in json.loads(labels_path.read_text(encoding="utf-8")):
            if rec["status"] in ("ok", "skipped_empty_cluster"):
                existing[rec["latent_id"]] = rec

    print(f"Labeling {run_dir.name} ({n_latents} latents, {len(existing)} already done) ...")
    records = []
    for latent_id in range(n_latents):
        if latent_id in existing:
            records.append(existing[latent_id])
            continue
        record = label_latent(
            latent_id, assignments, assignment_strength, sentences_df, n_top, n_random, seed, judge_model
        )
        records.append(record)
        print(f"  latent {latent_id}: {record['status']}" + (f" ({record['title']})" if record["title"] else ""))
        labels_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    labels_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return labels_path


def expand_layer_runs(layer: int, trained_sae_dir: Path) -> list[Path]:
    run_dirs = []
    for p in sorted(trained_sae_dir.glob(f"L{layer:02d}_*")):
        if p.is_dir() and (p / "config.json").exists():
            run_dirs.append(p)
    return run_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--trained-sae-dir", default=str(DEFAULT_TRAINED_SAE_DIR))
    parser.add_argument("--activations-dir", default=str(DEFAULT_ACTIVATIONS_DIR))
    parser.add_argument("--n-top", type=int, default=100)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
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

    sentences_df = pd.read_parquet(Path(args.activations_dir) / "sentences.parquet")
    if not (sentences_df["global_idx"].to_numpy() == np.arange(len(sentences_df))).all():
        raise RuntimeError("sentences.parquet rows are not in global_idx order; can't index by position.")

    for run_dir in run_dirs:
        label_run(
            run_dir, sentences_df, n_top=args.n_top, n_random=args.n_random,
            seed=args.seed, judge_model=args.judge_model, force=args.force,
        )
    print(f"Done: labeled {len(run_dirs)} run(s).")


if __name__ == "__main__":
    main()
