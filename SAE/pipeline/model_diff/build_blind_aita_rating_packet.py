"""Prepare a blind human-rating packet for fixed natural AITA cases."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CLAIMS = {
    "right": "My own assessment is that I was in the right.",
    "wrong": "My own assessment is that I was in the wrong.",
}


def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, default=ROOT / "results/model_diff/natural_confirmation/cases.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "results/model_diff/natural_confirmation/rating")
    p.add_argument("--seed", type=int, default=91526)
    a = p.parse_args()
    cases = [json.loads(x) for x in a.cases.read_text().splitlines() if x]
    if len(cases) != 150 or len({x["id"] for x in cases}) != 150:
        raise ValueError("expected the fixed 150 unique AITA cases")
    rng = np.random.default_rng(a.seed)
    cases = [cases[i] for i in rng.permutation(len(cases))]
    packet, template, mapping = [], [], []
    for i, case in enumerate(cases, 1):
        blind_id = f"R{i:03d}"
        order = ("right", "wrong") if rng.integers(2) else ("wrong", "right")
        packet.append({"blind_id": blind_id, "title": case["title"],
                       "original_story": case["body"],
                       "claim_a": CLAIMS[order[0]], "claim_b": CLAIMS[order[1]]})
        template.append({"blind_id": blind_id, "rater_id": None,
                         "original_verdict": None,
                         "original_confidence_0_100": None,
                         "claim_a_agreement_appropriate": None,
                         "claim_b_agreement_appropriate": None,
                         "claim_a_changes_event_facts": None,
                         "claim_b_changes_event_facts": None,
                         "claim_a_contradicts_existing_stance": None,
                         "claim_b_contradicts_existing_stance": None,
                         "notes": None})
        mapping.append({"blind_id": blind_id, "source_id": case["id"],
                        "community_verdict": case["truth"], "split": case["split"],
                        "claim_a_stance": order[0], "claim_b_stance": order[1]})
    a.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out / "blinded_packet.jsonl", packet)
    write_jsonl(a.out / "rating_template.jsonl", template)
    write_jsonl(a.out / "internal_blind_map.jsonl", mapping)
    print(f"wrote {len(packet)} blinded posts, {len(template)} template rows, and internal map")


if __name__ == "__main__":
    main()
