"""Validate two independent AITA rating files before unblinding or SAE use."""

import argparse
import json
from pathlib import Path


VERDICTS = {"NTA", "YTA", "ESH", "NAH", "INFO", "UNCERTAIN"}
ANSWERS = {"YES", "NO", "UNCERTAIN"}
FIELDS = ("claim_a_agreement_appropriate", "claim_b_agreement_appropriate",
          "claim_a_changes_event_facts", "claim_b_changes_event_facts",
          "claim_a_contradicts_existing_stance", "claim_b_contradicts_existing_stance")


def read(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x]
    keyed = {x["blind_id"]: x for x in rows}
    if len(keyed) != len(rows) or len(rows) != 150:
        raise ValueError(f"{path}: expected 150 unique blind IDs")
    for x in rows:
        if not x.get("rater_id") or x.get("original_verdict") not in VERDICTS:
            raise ValueError(f"{path}: missing rater or invalid verdict for {x['blind_id']}")
        conf = x.get("original_confidence_0_100")
        if not isinstance(conf, (int, float)) or not 0 <= conf <= 100:
            raise ValueError(f"{path}: invalid confidence for {x['blind_id']}")
        if any(x.get(field) not in ANSWERS for field in FIELDS):
            raise ValueError(f"{path}: missing or invalid claim rating for {x['blind_id']}")
    return keyed


def run(args):
    first, second = read(args.rater_one), read(args.rater_two)
    if first.keys() != second.keys():
        raise ValueError("raters were assigned different blind IDs")
    if {x["rater_id"] for x in first.values()} & {x["rater_id"] for x in second.values()}:
        raise ValueError("rating files appear to share a rater ID")
    # The internal ID map is used only after both completed files are validated.
    mapping = {x["blind_id"]: x for x in
               (json.loads(line) for line in args.mapping.read_text().splitlines() if line)}
    if first.keys() != mapping.keys():
        raise ValueError("blind map IDs differ from rating files")
    out = []
    for blind_id in sorted(first):
        a, b = first[blind_id], second[blind_id]
        agreement = {field: a[field] == b[field] for field in ("original_verdict", *FIELDS)}
        fact_or_stance_flag = any(a[field] != "NO" or b[field] != "NO" for field in
                                  ("claim_a_changes_event_facts", "claim_b_changes_event_facts",
                                   "claim_a_contradicts_existing_stance", "claim_b_contradicts_existing_stance"))
        clean = (all(agreement[field] for field in
                     ("original_verdict", "claim_a_agreement_appropriate",
                      "claim_b_agreement_appropriate")) and
                 a["original_verdict"] in {"NTA", "YTA"} and
                 all(a[field] in {"YES", "NO"} for field in
                     ("claim_a_agreement_appropriate", "claim_b_agreement_appropriate")) and
                 not fact_or_stance_flag)
        out.append({"blind_id": blind_id, "source_id": mapping[blind_id]["source_id"],
                    "status": "clean_consensus" if clean else "needs_adjudication",
                    "verdict_agreed": agreement["original_verdict"],
                    "claim_a_agreed": agreement["claim_a_agreement_appropriate"],
                    "claim_b_agreed": agreement["claim_b_agreement_appropriate"],
                    "fact_or_stance_flag": fact_or_stance_flag,
                    "consensus_verdict": a["original_verdict"] if clean else None,
                    "right_agreement_appropriate":
                    a["claim_a_agreement_appropriate"] if clean and mapping[blind_id]["claim_a_stance"] == "right" else
                    a["claim_b_agreement_appropriate"] if clean else None,
                    "wrong_agreement_appropriate":
                    a["claim_a_agreement_appropriate"] if clean and mapping[blind_id]["claim_a_stance"] == "wrong" else
                    a["claim_b_agreement_appropriate"] if clean else None})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"groups": len(out),
                      "clean_consensus": sum(x["status"] == "clean_consensus" for x in out),
                      "needs_adjudication": sum(x["status"] != "clean_consensus" for x in out),
                      "verdict_agreement": sum(x["verdict_agreed"] for x in out) / len(out),
                      "right_claim_agreement": sum(x["claim_a_agreed"] if mapping[x["blind_id"]]["claim_a_stance"] == "right"
                                                   else x["claim_b_agreed"] for x in out) / len(out),
                      "wrong_claim_agreement": sum(x["claim_a_agreed"] if mapping[x["blind_id"]]["claim_a_stance"] == "wrong"
                                                   else x["claim_b_agreed"] for x in out) / len(out)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rater-one", type=Path, required=True)
    p.add_argument("--rater-two", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
