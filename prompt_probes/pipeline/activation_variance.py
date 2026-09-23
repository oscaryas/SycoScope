#!/usr/bin/env python3
"""Balanced, multivariate system-by-question decomposition of cached activations.

Descriptive sums of squares in raw residual-stream coordinates, not probe scores,
causal fractions of response meaning, or independent-observation ANOVA p-values.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import common


COMPONENTS = ("system", "question", "interaction_and_generation_noise")


def decompose(x):
    """x has shape (system conditions, matched questions, activation dimensions)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or min(x.shape[:2]) < 2 or not np.isfinite(x).all():
        raise ValueError("Need finite balanced S x Q x D data, with S,Q >= 2")
    s, q, _ = x.shape
    grand = x.mean(axis=(0, 1))
    system = x.mean(axis=1) - grand
    question = x.mean(axis=0) - grand
    total = float(np.square(x - grand).sum())
    if total <= 0:
        raise ValueError("Zero total activation variance")
    system_by_condition = q * np.square(system).sum(axis=1)
    ss_system = float(system_by_condition.sum())
    ss_question = float(s * np.square(question).sum())
    residual = x - grand - system[:, None, :] - question[None, :, :]
    ss_residual = float(np.square(residual).sum())
    ss = dict(zip(COMPONENTS, (ss_system, ss_question, ss_residual)))
    np.testing.assert_allclose(sum(ss.values()), total, rtol=1e-10, atol=1e-8)
    return {
        "n_system_conditions": s, "n_questions": q, "hidden_size": x.shape[2],
        "total_ss": total, "ss": ss,
        "pct": {k: 100 * v / total for k, v in ss.items()},
        "condition_contribution_pct_of_total": (100 * system_by_condition / total).tolist(),
    }


def bootstrap_kernels(x):
    """Exact sufficient statistics for resampling whole questions, all systems intact.

    Small Q x Q Gram matrices avoid rebuilding a 40 x 400 x 4096 tensor on
    every draw. Float64 accumulation; no PCA, whitening, or approximation.
    """
    s, q, _ = x.shape
    question_mean = np.asarray(x, dtype=np.float64).mean(axis=0)
    g = question_mean - question_mean.mean(axis=0)
    question_gram = g @ g.T
    contrast_gram = np.zeros((q, q), dtype=np.float64)
    for condition in x:
        delta = np.asarray(condition, dtype=np.float64) - question_mean
        contrast_gram += delta @ delta.T
    return s, question_gram, contrast_gram


def resampled_shares(kernels, counts):
    s, g, k = kernels
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    q = counts.sum(axis=1)
    if np.any(counts < 0) or np.any(q < 2):
        raise ValueError("Invalid question multiplicities")
    system = np.einsum("bi,ij,bj->b", counts, k, counts, optimize=True) / q
    question = s * (counts @ np.diag(g)
                    - np.einsum("bi,ij,bj->b", counts, g, counts, optimize=True) / q)
    residual = counts @ np.diag(k) - system
    ss = np.column_stack([system, question, residual])
    if np.min(ss) < -1e-7 * max(1.0, np.max(np.abs(ss))):
        raise ValueError("Negative bootstrap sum of squares")
    ss = np.maximum(ss, 0)
    if np.any(ss.sum(axis=1) <= 0):
        raise ValueError("Zero bootstrap variance")
    return 100 * ss / ss.sum(axis=1, keepdims=True)


def question_bootstrap(x, n_boot=200, seed=20260916):
    q = x.shape[1]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(q, np.full(q, 1 / q), size=n_boot)
    kernels = bootstrap_kernels(x)
    original = resampled_shares(kernels, np.ones(q))[0]
    expected = list(decompose(x)["pct"].values())
    np.testing.assert_allclose(original, expected, atol=1e-7)
    draws = resampled_shares(kernels, counts)
    limits = np.quantile(draws, [.025, .975], axis=0)
    return {"n_boot": n_boot, "seed": seed,
            "method": "question-cluster percentile bootstrap; all systems retained together",
            "pct_95_interval": {k: limits[:, i].tolist() for i, k in enumerate(COMPONENTS)}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run, n_questions, seed):
    slugs = common.all_slugs(include_neutral=False)
    split_path = run / "prompt_split.json"
    test = json.loads(split_path.read_text())["test"]
    if len(set(test)) != len(test) or len(test) < n_questions:
        raise ValueError("Not enough distinct held-out questions")
    ids = sorted(test) if len(test) == n_questions else sorted(
        np.random.default_rng(seed).choice(sorted(test), n_questions, replace=False).tolist())
    maps, provenance, flags, conditions = {}, {}, {}, []
    clean = set(ids)
    for slug in slugs:
        index_path = run / "activations" / f"{slug}_index.jsonl"
        rows = common.read_jsonl(index_path)
        lookup = {}
        for i, r in enumerate(rows):
            key = (r["polarity"], r["prompt_id"])
            if key in lookup or r["row"] != i or r["slug"] != slug:
                raise ValueError(f"Invalid index mapping: {slug}/{key}")
            lookup[key] = i
        maps[slug] = {}
        for polarity in common.POLARITIES:
            name = slug + "/" + polarity
            ix = [lookup[polarity, q] for q in ids]
            selected = [rows[i] for i in ix]
            if any(r["n_response_tokens"] <= 0 or r.get("degenerate") == "empty" for r in selected):
                raise ValueError(f"Missing valid response span in {name}")
            maps[slug][polarity] = ix
            flags[name] = dict(Counter(r.get("degenerate") or "unflagged" for r in selected))
            clean -= {r["prompt_id"] for r in selected if r.get("degenerate") == "repetitive"}
            conditions.append(name)
        npz_path = run / "activations" / f"{slug}.npz"
        stat = npz_path.stat()
        provenance[slug] = {"index_sha256": sha(index_path), "index_rows": len(rows),
                            "npz_resolved_path": str(npz_path.resolve()),
                            "npz_bytes": stat.st_size, "npz_mtime_ns": stat.st_mtime_ns}
    return maps, {
        "run": str(run.resolve()), "dataset": "original Perez prompt pool",
        "model": "meta-llama/Llama-3.1-8B-Instruct", "position": "response",
        "layer_convention": "zero-based transformer block; hidden_states[layer + 1]",
        "selection": "frozen original probe holdout; no activation-based selection",
        "question_ids": ids, "n_questions": len(ids), "conditions": conditions,
        "n_responses": len(ids) * len(conditions),
        "nonrepetitive_complete_question_ids": sorted(clean), "flags": flags,
        "split_sha256": sha(split_path), "source_indices": provenance,
        "primary_filter": "all nonempty cached responses, including repetitive/refusal/short/truncated",
        "scaling": "raw activation coordinates, float64 sums of squares; no standardization",
        "neutral_condition": "absent; compares 40 inducing strings, not system versus no system",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_report(out, manifest, results):
    text = ["# System-prompt versus question activation variance", "",
            f"{manifest['n_questions']} matched held-out Perez questions × 40 system-prompt conditions "
            f"(20 pairs, both poles) = {manifest['n_responses']:,} cached responses. "
            "Llama-3.1-8B-Instruct; full-response-mean residual-stream activations.", "",
            "## Method", "",
            "At each layer: h[s,q] = grand mean + system mean deviation + question mean deviation "
            "+ interaction/residual. Sum squared Euclidean deviations across all 4096 raw activation "
            "coordinates. System SS = Q × sum_s ||mean_q(h[s,q]) − grand||²; question SS = "
            "S × sum_q ||mean_s(h[s,q]) − grand||². Residual SS is the squared norm left after "
            "subtracting both main effects. These sum to total SS in this balanced design.", "",
            "Intervals: 200 question-level bootstrap resamples, retaining all 40 conditions per "
            "sampled question and recomputing all means. They describe question-sampling uncertainty "
            "conditional on these fixed prompts and realized generations, not generation-seed uncertainty. "
            "No F-tests or ANOVA p-values are used.", "",
            "## Results (% of total activation variation)", "",
            "| Layer | System [95% interval] | Question [95% interval] | Interaction + noise [95% interval] |",
            "|---|---:|---:|---:|"]
    for key, r in results.items():
        cells = []
        for c in COMPONENTS:
            lo, hi = r["bootstrap"]["pct_95_interval"][c]
            cells.append(f"{r['primary']['pct'][c]:.2f} [{lo:.2f}, {hi:.2f}]")
        text.append(f"| {key} | " + " | ".join(cells) + " |")
    text += ["", "## Nonrepetitive complete-question sensitivity", "",
             f"Retain only the {len(manifest['nonrepetitive_complete_question_ids'])} questions with "
             "no repetitive flag in any condition. This is an outcome-selected sensitivity, not the "
             "primary estimate. Refusals, short answers and truncations are retained.", "",
             "| Layer | System % | Question % | Interaction + noise % |", "|---|---:|---:|---:|"]
    for key, r in results.items():
        vals = r["nonrepetitive_complete_questions"]["pct"]
        text.append(f"| {key} | " + " | ".join(f"{vals[c]:.2f}" for c in COMPONENTS) + " |")
    text += ["", "## Interpretation and limits", "",
             "- This is activation-variance attribution across the chosen conditions, not a percentage "
             "of response meaning, and not an estimate of presence-versus-absence of a system prompt. "
             "No neutral/no-system condition is present in this original cache. general_baseline is "
             "a substantive positive/negative prompt pair, not a neutral assistant.",
             "- Only one generation exists per system–question cell: system×question interaction, "
             "sampling noise and unmodeled generation effects cannot be separated. The system main "
             "effect alone is not the whole system-prompt effect.",
             "- Response text varies across conditions. Style, content and length changes contribute "
             "to these activations. Question effects include topic and other properties of the exact "
             "question, not just its semantic intent.",
             "- Raw-coordinate variance weights high-variance activation directions more heavily. "
             "This descriptive metric does not establish behaviorally important or causal dimensions. "
             "Percentages have a separate denominator at each layer.",
             "- results.json also contains per-pair two-condition decompositions and each condition's "
             "contribution to the global system sum of squares. Per-pair percentages have different "
             "denominators; they are not an additive breakdown of the global percentages.",
             "- This is the original Perez dataset, not the still-running matched-backend Perez/Dolly "
             "dataset-control experiment. No probe fitting, generation, GPU use or API calls were needed.", ""]
    (out / "RESULTS.md").write_text("\n".join(text))
    (out / "results.json").write_text(json.dumps({"manifest": manifest, "layers": results}, indent=2) + "\n")


def plot(out, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = [int(k.rsplit("L", 1)[1]) for k in results]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for component, label, color in zip(COMPONENTS,
            ("System-prompt main effect", "Question main effect", "Interaction + generation noise"),
            ("#2563eb", "#0f766e", "#b45309")):
        values = [r["primary"]["pct"][component] for r in results.values()]
        bounds = np.array([r["bootstrap"]["pct_95_interval"][component] for r in results.values()])
        ax.plot(layers, values, "o-", color=color, label=label)
        ax.fill_between(layers, bounds[:, 0], bounds[:, 1], color=color, alpha=.16)
    ax.set(xlabel="Transformer block (zero-based)", ylabel="Share of raw activation variation (%)",
           ylim=(0, 100), xticks=layers,
           title="400 matched questions × 40 system conditions\nFull-response-mean activations · original Perez dataset")
    ax.grid(alpha=.2)
    ax.legend(loc="best")
    fig.text(.5, .015, "Bands: 95% question-bootstrap intervals. Residual includes interaction and sampling noise; no neutral condition.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out / "variance_by_layer.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="llama31_5k_subset")
    parser.add_argument("--n-questions", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    run = common.resolve_run_dir(args.run_name, create=False)
    out = run / "analysis" / f"activation_variance_{args.n_questions}"
    out.mkdir(parents=True, exist_ok=True)
    maps, manifest = prepare(run, args.n_questions, args.seed)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {manifest['n_responses']} responses; {len(manifest['nonrepetitive_complete_question_ids'])} clean questions", flush=True)
    clean = np.array([q in set(manifest["nonrepetitive_complete_question_ids"]) for q in manifest["question_ids"]])
    results = {}
    for layer in (0, 4, 8, 12, 16, 20, 24, 28):
        key = f"response_L{layer:02d}"
        x = None
        for j, (slug, by_polarity) in enumerate(maps.items()):
            with np.load(run / "activations" / f"{slug}.npz") as arrays:
                array = arrays[key]
            if array.shape[0] != manifest["source_indices"][slug]["index_rows"]:
                raise ValueError(f"Activation/index mismatch for {slug}")
            if x is None:
                x = np.empty((len(manifest["conditions"]), args.n_questions, array.shape[1]), dtype=np.float32)
            for k, polarity in enumerate(common.POLARITIES):
                x[2 * j + k] = array[by_polarity[polarity]]
            del array
        print(f"{key}: loaded matched activations; computing decomposition and bootstrap", flush=True)
        primary = decompose(x)
        results[key] = {
            "primary": primary, "bootstrap": question_bootstrap(x, seed=args.seed),
            "nonrepetitive_complete_questions": decompose(x[:, clean]),
            "per_prompt_pair": {slug: decompose(x[2*j:2*j+2]) for j, slug in enumerate(maps)},
        }
        del x
        write_report(out, manifest, results)
        print(f"{key}: {primary['pct']}", flush=True)
    plot(out, results)
    (out / "complete.json").write_text(json.dumps({"layers": list(results), "n_responses": manifest["n_responses"]}) + "\n")
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
