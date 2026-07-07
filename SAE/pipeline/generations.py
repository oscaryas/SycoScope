"""
Generate Llama-3-8B-Instruct responses for every prompt in SAE/datasets.
Writes one JSONL file per dataset to SAE/datasets/generations/.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import datasets, inference
from utils.model import DEFAULT_MODEL, cleanup, load_model_and_tokenizer

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "generations"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None, help="Cap prompts per dataset (for smoke testing)")
    parser.add_argument(
        "--n-samples", type=int, default=1,
        help="Sampled generations per prompt (uses sampling; each gets a distinct sample_idx)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    for filename in datasets.list_dataset_files():
        dataset_name = Path(filename).stem
        records = list(datasets.iter_prompts(filename))
        if args.limit:
            records = records[: args.limit]
        records = [
            {**record, "sample_idx": i}
            for record in records
            for i in range(args.n_samples)
        ]
        print(f"{filename}: generating {len(records)} responses ({args.n_samples} sample(s)/prompt)")

        out_path = output_dir / f"{dataset_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f, tqdm(
            total=len(records), desc=dataset_name, unit="response"
        ) as progress:
            for batch in inference.iter_batches(records, args.batch_size):
                responses = inference.generate_batch(
                    model,
                    tokenizer,
                    [r["text"] for r in batch],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                for record, response in zip(batch, responses):
                    f.write(
                        json.dumps(
                            {
                                "dataset": record["dataset"],
                                "row_id": record["row_id"],
                                "prompt_col": record["prompt_col"],
                                "prompt": record["text"],
                                "sample_idx": record["sample_idx"],
                                "response": response,
                                "model": args.model,
                            }
                        )
                        + "\n"
                    )
                progress.update(len(batch))
        print(f"Wrote {out_path}")

    cleanup(model, tokenizer)


if __name__ == "__main__":
    main()
