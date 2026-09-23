"""CPU-only rebuild of the frozen SAE sentence index; never loads LLM weights.

Uses the original cache_activations segmentation helper and source-file order.
The original spaCy package is absent here, so the English tokenizer plus
sentencizer is reconstructed with spacy.blank("en"). Joining stored SAE
assignments remains provisional until the row count and spot checks pass.
"""
import argparse
import json
import sys
from pathlib import Path


EXPECTED_SOURCE_HASH = "750d3a8e2c2be6e3c03aa3ecc55812432c587acbdc1b2e54da51eeb38719a9f4"
EXPECTED_ASSIGNMENT_ROWS = 833300


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("SAE/results"))
    p.add_argument("--out-dir", type=Path, default=Path("SAE/results/model_diff/sae_index_rebuild"))
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--max-length", type=int, default=8192)
    a = p.parse_args()
    import pandas as pd
    import spacy
    from transformers import AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from SAE.pipeline.cache_activations import (
        build_sentence_table, hash_input_files, iter_input_records)

    source_hash = hash_input_files(a.input)
    if source_hash != EXPECTED_SOURCE_HASH:
        raise ValueError(f"source files changed: {source_hash}")
    t = AutoTokenizer.from_pretrained(a.model, local_files_only=True, use_fast=True)
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    records = list(iter_input_records(a.input))
    skip_log = []
    sentence_rows, response_rows = build_sentence_table(
        records, t, nlp, a.max_length, skip_log)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sentence_rows).to_parquet(a.out_dir / "sentences.parquet", index=False)
    pd.DataFrame(response_rows).to_parquet(a.out_dir / "responses.parquet", index=False)
    meta = {"source_hash": source_hash, "n_input_records": len(records),
            "n_sentences": len(sentence_rows), "n_responses": len(response_rows),
            "n_skip_events": len(skip_log), "max_length": a.max_length,
            "spacy_version": spacy.__version__, "spacy_pipeline": nlp.pipe_names,
            "tokenizer_class": type(t).__name__,
            "matches_stored_assignment_count": len(sentence_rows) == EXPECTED_ASSIGNMENT_ROWS}
    (a.out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (a.out_dir / "skipped.jsonl").open("w") as f:
        for event in skip_log:
            f.write(json.dumps(event) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
