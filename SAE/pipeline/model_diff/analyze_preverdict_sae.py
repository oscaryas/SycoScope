"""Audit frozen chat SAEs on matched natural AITA decision states."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


FORMATS = ("yta_nta", "nta_yta", "four_yta_last", "four_nta_last")
SAES = ("L14_n15_k3_uncentered_s0", "L14_n40_k3_uncentered_s0",
        "L23_n30_k3_uncentered_s0")
SELECTED = {SAES[0]: (14,), SAES[1]: (16, 17, 39), SAES[2]: (13, 15)}


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def quantiles(x):
    a = np.asarray(x, dtype=np.float64)
    return {"median": round(float(np.median(a)), 4),
            "q10": round(float(np.quantile(a, .1)), 4),
            "q90": round(float(np.quantile(a, .9)), 4), "n": len(a)}


def load_sae(root, name):
    path = root / name
    cfg = json.loads((path / "config.json").read_text())
    weights = torch.load(path / "sae.pt", map_location="cpu", weights_only=True)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    w = {k: x.float().numpy() for k, x in weights.items()}
    labels = json.loads((path / "labels.json").read_text())
    return {"config": cfg, "weights": w,
            "titles": {x["latent_id"]: x.get("title") for x in labels},
            "path": path}


def encode(states, sae):
    x = np.asarray(states, dtype=np.float32)
    w = sae["weights"]
    z = np.maximum(0, (x - w["b_dec"]) @ w["W_enc"].T + w["b_enc"])
    k = sae["config"]["k"]
    idx = np.argpartition(z, -k, axis=1)[:, -k:]
    val = np.take_along_axis(z, idx, axis=1)
    sparse = np.zeros_like(z)
    np.put_along_axis(sparse, idx, val, axis=1)
    rec = sparse @ w["W_dec"].T + w["b_dec"]
    norm = np.linalg.norm(x, axis=1)
    err = np.linalg.norm(x - rec, axis=1) / np.maximum(norm, 1e-8)
    top1 = z.argmax(1)
    top1[z.max(1) <= .01] = -1
    return {"codes": sparse, "top1": top1,
            "relative_error": err, "state_norm": norm,
            "max_strength": z.max(1)}


def reference_states(directory, n_files):
    files = sorted(directory.glob("*.npz"))[:n_files]
    if len(files) != n_files:
        raise ValueError(f"need {n_files} old chat state files, got {len(files)}")
    sentences, prompt_ends = [], []
    for path in files:
        with np.load(path) as x:
            sentences.append(x["sentence_states"])
            prompt_ends.append(x["pre_response_state"])
    return np.concatenate(sentences), np.stack(prompt_ends), len(files)


def score_rows(root, model):
    rows = jsonl(root / ("chat.jsonl" if model == "chat" else "base.jsonl"))
    rows += jsonl(root / ("format_chat.jsonl" if model == "chat" else "format_base.jsonl"))
    rows = [x for x in rows if x["example_order"] in FORMATS]
    keyed = {(x["response_id"], x["example_order"]): x for x in rows}
    if len(keyed) != 1800 or len(rows) != 1800:
        raise ValueError(f"expected 1,800 unique scored {model} contexts")
    return keyed


def verify(meta, states, scores, model):
    if len(meta) != 1800 or states.shape != (1800, 2, 2, 4096):
        raise ValueError(f"incomplete {model} hidden-state extraction")
    if states.dtype != np.float16 or not np.isfinite(states).all():
        raise ValueError(f"invalid {model} states")
    by_row = sorted(meta, key=lambda x: x["state_row"])
    if [x["state_row"] for x in by_row] != list(range(1800)):
        raise ValueError(f"duplicate/gapped {model} state rows")
    for x in by_row:
        key = (x["response_id"], x["example_order"])
        if key not in scores or x["prompt_ids_sha256"] != scores[key]["prompt_ids_sha256"]:
            raise ValueError(f"scored prompt hash mismatch: {model} {key}")
        if x["prompt_tokens"] != scores[key]["prompt_tokens"]:
            raise ValueError(f"scored token count mismatch: {model} {key}")
        if x["layers"] != [14, 23] or x["positions"] != ["story_end", "pre_verdict"]:
            raise ValueError(f"unexpected layer/position layout: {model}")
    return by_row


def code_summary(enc, sae, training_assignments=None):
    sparse = enc["codes"]
    counts = Counter(int(x) for x in enc["top1"] if x >= 0)
    result = {"relative_reconstruction_error": quantiles(enc["relative_error"]),
              "hidden_state_norm": quantiles(enc["state_norm"]),
              "max_activation": quantiles(enc["max_strength"]),
              "zero_code_fraction": round(float(np.mean(enc["max_strength"] <= .01)), 4),
              "top1": {str(i): {"count": n, "title": sae["titles"].get(i)}
                       for i, n in counts.most_common(10)},
              "selected_features": {}}
    for latent in SELECTED[sae["path"].name]:
        z = sparse[:, latent]
        result["selected_features"][str(latent)] = {
            "title": sae["titles"].get(latent),
            "positive_rate": round(float(np.mean(z > .01)), 4),
            "strength": quantiles(z)}
    if training_assignments is not None:
        result["training_sentence_topk_positive"] = training_assignments
    return result


def old_assignment_reference(sae):
    ids = np.load(sae["path"] / "assignments_topk.npy", mmap_mode="r")
    strength = np.load(sae["path"] / "assignments_topk_strength.npy", mmap_mode="r")
    result = {}
    for latent in SELECTED[sae["path"].name]:
        active = np.any((ids == latent) & (strength > .01), axis=1)
        result[str(latent)] = round(float(active.mean()), 4)
    return result


def code_stability(enc_story, enc_pre, meta):
    index = {(x["response_id"], x["example_order"]): x["state_row"] for x in meta}
    rids = sorted({x["response_id"] for x in meta})
    out = {"story_end_to_pre_verdict_top1_change":
           round(float(np.mean(enc_story["top1"] != enc_pre["top1"])), 4)}
    for a, b in ((FORMATS[0], FORMATS[1]), (FORMATS[2], FORMATS[3])):
        out[f"{a}_vs_{b}_top1_change"] = round(float(np.mean([
            enc_pre["top1"][index[(rid, a)]] != enc_pre["top1"][index[(rid, b)]]
            for rid in rids])), 4)
    return out


def feature_behavior(enc, meta, scores, sae):
    index = {(x["pair_id"], x["stance"], x["example_order"]): x["state_row"] for x in meta}
    truth = {x["pair_id"]: x["truth"] for x in meta}
    groups = sorted(truth)
    result = {}
    for fmt in FORMATS:
        for label in ("NTA", "YTA"):
            ids = [g for g in groups if truth[g] == label]
            stance = "wrong" if label == "NTA" else "right"
            row_ids = [index[(g, stance, fmt)] for g in ids]
            flags = np.array([scores[(meta[i]["response_id"], fmt)]["candidate_choice"] ==
                              scores[(meta[i]["response_id"], fmt)]["asserted"] for i in row_ids])
            cell = {"n_groups": len(ids), "false_agreement_groups": int(flags.sum()), "features": {}}
            for latent in SELECTED[sae["path"].name]:
                z = enc["codes"][row_ids, latent] > .01
                cell["features"][str(latent)] = {
                    "positive_false_agreement": int((z & flags).sum()),
                    "positive_correction": int((z & ~flags).sum())}
            result[f"{label}/{fmt}"] = cell
    return result


def run(args):
    root = args.root
    bmeta = jsonl(root / "preverdict/preverdict_base.jsonl")
    cmeta = jsonl(root / "preverdict/preverdict_chat.jsonl")
    bstates = np.load(root / "preverdict/preverdict_base.npy", mmap_mode="r")
    cstates = np.load(root / "preverdict/preverdict_chat.npy", mmap_mode="r")
    bscores, cscores = score_rows(root, "base"), score_rows(root, "chat")
    bmeta = verify(bmeta, bstates, bscores, "base")
    cmeta = verify(cmeta, cstates, cscores, "chat")
    for a, b in zip(bmeta, cmeta):
        for field in ("response_id", "pair_id", "stance", "truth", "example_order",
                      "prompt_ids_sha256", "story_end_token", "pre_verdict_token"):
            if a[field] != b[field]:
                raise ValueError(f"base/chat {field} mismatch at {a['state_row']}")
    report = {"groups": 150, "contexts_per_model": 1800,
              "base_chat_prompt_hashes_match": True,
              "state_hashes_match_prior_scores": True,
              "crossing_story_end_tokens": sum(x["crosses_story_end"] for x in cmeta),
              "raw_state_cosine": {}, "saes": {}}
    for j, layer in enumerate((14, 23)):
        for k, position in enumerate(("story_end", "pre_verdict")):
            b = bstates[:, j, k].astype(np.float32)
            c = cstates[:, j, k].astype(np.float32)
            cosine = np.sum(b*c, axis=1) / np.maximum(np.linalg.norm(b, axis=1)*np.linalg.norm(c, axis=1), 1e-8)
            report["raw_state_cosine"][f"L{layer}/{position}"] = quantiles(cosine)
    ref_sentence, ref_prompt, n_files = reference_states(args.reference_dir, args.reference_files)
    report["reference"] = {"old_chat_teacher_forced_response_files": n_files,
                           "response_sentence_states": len(ref_sentence),
                           "old_chat_template_prompt_ends": len(ref_prompt)}
    for name in SAES:
        sae = load_sae(args.sae_root, name)
        layer_idx = (14, 23).index(sae["config"]["layer"])
        old_assignments = old_assignment_reference(sae)
        story = encode(cstates[:, layer_idx, 0], sae)
        pre = encode(cstates[:, layer_idx, 1], sae)
        item = {"layer": sae["config"]["layer"],
                "n_latents": sae["config"]["n_latents"],
                "training_topk_positive_rates": old_assignments,
                "natural_story_end": code_summary(story, sae),
                "natural_pre_verdict": code_summary(pre, sae),
                "code_stability": code_stability(story, pre, cmeta),
                "unwarranted_stance_association": feature_behavior(pre, cmeta, cscores, sae)}
        if sae["config"]["layer"] == 14:
            item["old_response_sentence_reference"] = code_summary(encode(ref_sentence, sae), sae)
            item["old_chat_template_prompt_end_reference"] = code_summary(encode(ref_prompt, sae), sae)
        report["saes"][name] = item
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("paired contexts", report["contexts_per_model"], "reference sentences", len(ref_sentence))
    for name in SAES:
        item = report["saes"][name]
        print(name, "preverdict error", item["natural_pre_verdict"]["relative_reconstruction_error"],
              "top1", list(item["natural_pre_verdict"]["top1"].items())[:4],
              "cue-change", item["code_stability"]["story_end_to_pre_verdict_top1_change"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("SAE/results/model_diff/natural_confirmation"))
    p.add_argument("--sae-root", type=Path, default=Path("SAE/results/trained_sae"))
    p.add_argument("--reference-dir", type=Path, default=Path("SAE/results/model_diff/grouped_600_results/chat_states"))
    p.add_argument("--reference-files", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("SAE/results/model_diff/natural_confirmation/preverdict/sae_audit.json"))
    run(p.parse_args())
