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
judge), oeq/ss/aita_yta (ELEPHANT social-sycophancy judge, run separately via
run_social_sycophancy_judge_oeq.py), aita_nta_flip (ELEPHANT moral-sycophancy
judge, run separately via run_moral_sycophancy_judge_aita.py --mode flip --
see social_generate.py / moral_generate.py, which only generate; judging is
not wired into this driver since the judge runners have their own --resume).
dissociating_sycophancy is NOT a generation task (it judges the source
dataset's fixed conversations) and syconbench is fetched, so neither applies
here.

Thinking models get --max-new-tokens 2048 (reasoning routinely runs past the
Llama-era 200-300 defaults; ActivationSteerer warns if any generation still
hits the cap). Nemotron gets --system-prompt 'detailed thinking on' -- its
reasoning is OFF otherwise. Row caps (--n) keep the whole matrix inside one
Colab session; rerun with a larger --n later and the checkpoints resume.

By default models run one at a time, each processing its own datasets
sequentially (a single GPU process, one generation script running at once) --
this is the safe default because a single thinking model at a reasonable
batch size already sits near a single A100-80GB's memory ceiling (Qwen3-8B
batch 64 peaks ~73GB, Qwen3-14B batch 32 peaks ~77.5GB per prior session
notes). --parallel-models N runs up to N models' dataset queues concurrently
in separate OS processes on the same GPU -- only safe when each model's own
--batch-scale is turned down enough that the sum of their peak memory stays
under the card's ceiling (e.g. two 8B-class models at batch 16, ~26GB each,
comfortably fit two-at-a-time on an 80GB card). Datasets WITHIN one model
still run sequentially -- only different models' queues overlap.
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
    ("Qwen/Qwen3.8-27B", None, 2),  # 64 layers x head_dim 256 KV + 54GB weights: keep small
]

DATASETS = [
    # (name, script, extra args)
    ("are_you_sure_mc", "are_you_sure_mc_generate.py", []),
    ("are_you_sure_freeform", "are_you_sure_freeform_generate.py", []),
    ("truthfulqa", "truthfulqa_sycophancyeval_generate.py", []),
    ("sypr", "sypr_praise_full_generate.py", []),
    ("oeq", "social_generate.py", ["--dataset", "oeq"]),
    ("ss", "social_generate.py", ["--dataset", "ss"]),
    ("aita_yta", "social_generate.py", ["--dataset", "aita_yta"]),
    ("aita_nta_flip", "moral_generate.py", ["--dataset", "aita_nta_flip"]),
]


def slug(model: str) -> str:
    return model.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, default=",".join(m for m, _, _ in MODELS))
    parser.add_argument("--datasets", type=str, default=",".join(d for d, _, _ in DATASETS))
    parser.add_argument("--n", type=int, default=1000, help="Row cap per dataset per model (0 = full).")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-scale", type=int, default=1,
                        help="Multiply each model's generation batch size (decode is memory-bound; "
                             "A100-80GB uses ~26GB at Qwen3-8B x16, so 4 is safe there -- turn this "
                             "DOWN when using --parallel-models > 1, since peak memory is per model "
                             "and multiple models hold weights + KV cache on the GPU at once).")
    parser.add_argument("--parallel-models", type=int, default=1,
                        help="Run up to N models' dataset queues concurrently (separate processes, "
                             "same GPU). Default 1 = fully sequential, the safe default. Only raise "
                             "this if --batch-scale has been sized so N models' peak memory fits the "
                             "card together -- an OOM in one process can still take down others "
                             "sharing the GPU.")
    parser.add_argument("--tar", type=str, default="/content/multimodel_generations.tar.gz")
    args = parser.parse_args()

    want_models = [m for m in MODELS if m[0] in set(args.models.split(","))]
    want_datasets = set(args.datasets.split(","))
    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def run_model_queue(model_spec):
        model, system_prompt, batch_size = model_spec
        for name, script, extra in DATASETS:
            if name not in want_datasets:
                continue
            out = GEN_DIR / slug(model) / name / "checkpoint.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(HERE / script),
                "--model", model, "--out", str(out),
                "--max-new-tokens", str(args.max_new_tokens),
                "--generation-batch-size", str(batch_size * args.batch_scale),
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
        # free GPU memory between datasets is handled by each script's cleanup on exit

    if args.parallel_models > 1:
        log(f"Running {len(want_models)} model queue(s) with up to {args.parallel_models} concurrent.")
        with ThreadPoolExecutor(max_workers=args.parallel_models) as pool:
            # subprocess.call blocks the calling thread but not the GIL, so N
            # threads here really do run N model queues' subprocesses at once.
            list(pool.map(run_model_queue, want_models))
    else:
        for model_spec in want_models:
            run_model_queue(model_spec)

    if args.tar:
        subprocess.call(["tar", "-czf", args.tar, "-C", str(GEN_DIR.parent), "multimodel"])
        log(f"tarred results to {args.tar}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
