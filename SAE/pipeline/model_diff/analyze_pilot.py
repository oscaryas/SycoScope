"""Descriptive paired-score and frozen-SAE summary for the tiny AITA pilot."""
import argparse
import json
from pathlib import Path
from statistics import mean


def rows(path):
    return {x["response_id"]: x for x in (json.loads(line) for line in Path(path).read_text().splitlines())}


def analyze(chat_path, base_path, labels_path):
    chat, base = rows(chat_path), rows(base_path)
    paired = []
    for rid in sorted(chat.keys() & base.keys()):
        a, b = chat[rid], base[rid]
        if (a["token_ids_sha256"], a["response_start"], a["n_tokens"]) != (
                b["token_ids_sha256"], b["response_start"], b["n_tokens"]):
            raise ValueError("token mismatch: " + rid)
        feat = [max((s["sae_activations"][j] for s in a["sentences"]), default=0)
                for j in range(len(a["sentences"][0]["sae_activations"]))] if a["sentences"] else []
        paired.append({"response_id": rid, "pair_id": a["pair_id"], "cell": a["cell"],
                       "side": a["side"], "verdict": a["verdict"],
                       "chat_minus_base_logprob": a["mean_forced_logprob"]-b["mean_forced_logprob"],
                       "features_max": feat})
    if not paired:
        raise ValueError("no paired responses")
    labels = {x["latent_id"]: x["title"] for x in json.loads(Path(labels_path).read_text())}
    cells = {c: [x for x in paired if x["cell"] == c] for c in ("both_nta", "mixed")}
    nta = {c: [x for x in group if x["verdict"] == "NTA"] for c, group in cells.items()}
    deltas = []
    if all(nta.values()):
        for j in range(len(nta["both_nta"][0]["features_max"])):
            positive = mean(x["features_max"][j] for x in nta["both_nta"])
            mixed = mean(x["features_max"][j] for x in nta["mixed"])
            deltas.append({"latent_id": j, "label": labels.get(j, ""),
                           "both_nta_mean_max": round(positive, 3),
                           "mixed_nta_mean_max": round(mixed, 3),
                           "difference": round(positive-mixed, 3)})
        deltas.sort(key=lambda x: abs(x["difference"]), reverse=True)
    return {"n_responses": len(paired), "n_conflicts": len({x["pair_id"] for x in paired}),
            "cell_mean_chat_minus_base_logprob": {
                c: round(mean(x["chat_minus_base_logprob"] for x in group), 4)
                if group else None for c, group in cells.items()},
            "nta_only_cell_mean_chat_minus_base_logprob": {
                c: round(mean(x["chat_minus_base_logprob"] for x in group), 4)
                if group else None for c, group in nta.items()},
            "nta_only_feature_max_differences": deltas,
            "paired_responses": [{k: v for k, v in x.items() if k != "features_max"} for x in paired],
            "interpretation_limit": "Eight exploratory conflict pairs from the SAE training corpus; no causal or held-out claim."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chat", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = analyze(a.chat, a.base, a.labels)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print("paired", report["n_responses"], "conflicts", report["n_conflicts"])
    print("cell means", report["cell_mean_chat_minus_base_logprob"])


if __name__ == "__main__":
    main()
