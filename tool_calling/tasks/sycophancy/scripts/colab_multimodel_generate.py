#!/usr/bin/env python3
"""
Colab driver: regenerate the sycophancy datasets with several models, using
the real generation scripts (so labels/judging/checkpointing are identical
to the Llama-3 originals), then tar the results for transfer.

Run detached from a notebook cell AFTER the kernel has exported HF_TOKEN /
ANTHROPIC_API_KEY into os.environ (google.colab.userdata):

    !cd /content/SycoScope && nohup python tool_calling/tasks/sycophancy/scripts/colab_multimodel_generate.py \
        > /content/gen.log 2>&1 &

Per model x dataset, output lands in
results/generations/multimodel/<model_slug>/<dataset>/checkpoint.jsonl
(the scripts' own resumable checkpoint format), so a killed run just picks
up where it left off on rerun.

Datasets: are_you_sure_mc (mechanical labels), are_you_sure_freeform
(Claude correctness judge), truthfulqa (Claude judge), sypr (Claude praise
judge). dissociating_sycophancy is NOT a generation task (it judges the
source dataset's fixed conversations) and syconbench is fetched, so neither
applies here; moral/social have no in-repo generation scripts.

Thinking models get --max-new-tokens 2048 (reasoning routinely runs past the
Llama-era 200-300 defaults; ActivationSteerer warns if any generation still
hits the cap). Nemotron gets --system-prompt 'detailed thinking on' -- its
reasoning is OFF otherwise. Row caps (--n) keep the whole matrix inside one
Colab session; rerun with a larger --n later and the checkpoints resume.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYCOPHANCY_DIR = HERE.parent
GEN_DIR = SYCOPHANCY_DIR / "results" / "generations" / "multimodel"

MODELS = [
    # (hf id, system prompt, generation batch size)
    ("Qwen/Qwen3-8B", None, 16),
    ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "detailed thinking on", 16),
    ("google/gemma-4-12B-it", None, 16),
    ("Qwen/Qwen3-14B", None, 16),
    ("Qwen/Qwen3.8-27B", None, 8),
]

DATASETS = [
    # (name, script, extra args)
    ("are_you_sure_mc", "are_you_sure_mc_generate.py", []),
    ("are_you_sure_freeform", "are_you_sure_freeform_generate.py", []),
    ("truthfulqa", "truthfulqa_sycophancyeval_generate.py", []),
    ("sypr", "sypr_praise_full_generate.py", []),
]


def slug(model: str) -> str:
    return model.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, default=",".join(m for m, _, _ in MODELS))
    parser.add_argument("--datasets", type=str, default=",".join(d for d, _, _ in DATASETS))
    parser.add_argument("--n", type=int, default=1000, help="Row cap per dataset per model (0 = full).")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--tar", type=str, default="/content/multimodel_generations.tar.gz")
    args = parser.parse_args()

    want_models = set(args.models.split(","))
    want_datasets = set(args.datasets.split(","))
    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    for model, system_prompt, batch_size in MODELS:
        if model not in want_models:
            continue
        for name, script, extra in DATASETS:
            if name not in want_datasets:
                continue
            out = GEN_DIR / slug(model) / name / "checkpoint.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(HERE / script),
                "--model", model, "--out", str(out),
                "--max-new-tokens", str(args.max_new_tokens),
                "--generation-batch-size", str(batch_size),
                *extra,
            ]
            if args.n:
                cmd += ["--n", str(args.n)]
            if system_prompt:
                cmd += ["--system-prompt", system_prompt]
            log(f"START {model} / {name}: {' '.join(cmd)}")
            t0 = time.time()
            rc = subprocess.call(cmd, cwd=str(HERE))
            log(f"END   {model} / {name}: rc={rc} in {(time.time() - t0) / 60:.1f} min")
        # free GPU memory between models is handled by each script's cleanup on exit

    if args.tar:
        subprocess.call(["tar", "-czf", args.tar, "-C", str(GEN_DIR.parent), "multimodel"])
        log(f"tarred results to {args.tar}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
