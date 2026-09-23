# Running SycoScope generation on Compute Canada Trillium

Trillium-specific setup and operation. Deployment plumbing only -- `social_generate.py` /
`moral_generate.py` and the judge scripts are unchanged; see `hpc/README.md` for the CSCS Clariden
side and the full shared design rationale (this doc only covers what's Trillium-specific).

## Cluster facts

| | |
|---|---|
| GPU | H100 SXM 80GB, 4/node |
| GPU allocation granularity | **1 GPU or a full node (4) only -- no 2/3** |
| Container tooling | Apptainer (`module load apptainer`); `.sif` must be built **off-cluster** |
| Compute-node internet | **None** -- confirmed explicitly in Trillium's own docs (login nodes have internet, compute nodes categorically don't) |
| Wall-clock | `compute` partition: 24h max, 15min minimum. `debug`: 2h (1 GPU) / 30min (8 GPU) |
| Account flag | `--account=<rrg-/rpp-/def->` (Alliance account string) |
| Persistent storage | `$PROJECT` (1TB default, backed up) -- **not** `$SCRATCH` (25TB, no fixed purge cadence yet, but documented as ephemeral) |
| Job arrays | Fully supported, with `%` concurrency throttling (`--array=1-12%4`) -- not used here (12 jobs submitted individually via `submit_matrix.sh`), but available if the matrix grows |
| Auto-requeue | Not offered/documented -- self-resubmission is Trillium's own documented pattern for jobs that outrun the wall-clock, which is what `submit_generate.sbatch` does automatically |

Because compute nodes have zero internet, weight prefetch (step 3 below) and the `.sif` transfer
(step 2) are not optional the way they might be elsewhere -- a generation job here has no fallback
path to fetch anything mid-run.

## One-time setup

### 1. Build and publish the container image

Same image as CSCS -- built once from `hpc/Dockerfile`, pushed to a registry:

```bash
docker build -t <registry>/sycoscope:latest -f hpc/Dockerfile .
docker push <registry>/sycoscope:latest
```

### 2. Build the `.sif` off-cluster and transfer it

Trillium can't pull from a registry itself, so convert the image to Apptainer format somewhere
with internet access (your laptop, a CI runner, or `apptainer build --remote` if you don't have
local root) and copy it onto `$PROJECT`:

```bash
apptainer build sycoscope.sif docker://<registry>/sycoscope:latest
scp sycoscope.sif trillium:$PROJECT/sycoscope.sif
```

Re-run this whenever `hpc/Dockerfile` changes -- there's no auto-pull/cache-refresh on this
cluster the way CSCS's Container Engine has.

### 3. Clone the repo onto `$PROJECT`

```bash
cd $PROJECT
git clone <this-repo-url> sycoscope-repo
```

Not `$SCRATCH` -- checkpoints from a job that self-resubmits across many hours (or days, for a
large `--n`) need to survive between submissions, and `$SCRATCH` has no purge guarantee but is
explicitly documented as not meant for anything you can't afford to lose.

Replace `CHANGE_ME_ACCOUNT` in `hpc/trillium/submit_generate.sbatch` with your actual Alliance
account string (`rrg-...` / `rpp-...` / `def-...`) before submitting anything.

### 4. Secrets

```bash
cat > ~/.sycoscope_secrets.env <<'EOF'
export HF_TOKEN=hf_...
export ANTHROPIC_API_KEY=sk-ant-...
EOF
chmod 600 ~/.sycoscope_secrets.env
```

`submit_generate.sbatch` sources this at the top of every job. Never commit it, never `cat`/`echo`
it.

### 5. Prefetch model weights (mandatory here, not optional)

On the **login node** (plain download, no GPU/allocation needed):

```bash
export HF_HOME=$PROJECT/huggingface
source ~/.sycoscope_secrets.env
python hpc/prefetch_weights.py
```

`submit_generate.sbatch` sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` and will fail fast if a
model wasn't staged here first -- there is no network fallback on Trillium's compute nodes.

## Submitting jobs

### Smoke test first

```bash
sbatch --time=00:15:00 --export=ALL,MODEL="Qwen/Qwen3-8B",DATASET=ss,N=8,BATCH_SIZE=4 \
    hpc/trillium/submit_generate.sbatch
```

Confirm it produces `checkpoint.jsonl` and (at this small `--n`) `summary.json`. Then test the
resume path: `scancel` mid-run, resubmit the identical command, confirm it picks up from the row
count already in `checkpoint.jsonl` instead of restarting.

### Full matrix

```bash
cd $PROJECT/sycoscope-repo
hpc/submit_matrix.sh trillium
```

Defaults: `--n 2000`, `--generation-batch-size 16`. Override via env vars, e.g.
`N=1000 BATCH_SIZE=8 hpc/submit_matrix.sh trillium`.

### A real memory lesson from this session, worth heeding here too

Running gemma-4-12B-it's generation at `--generation-batch-size 16` / `--max-new-tokens 2048` on a
40GB A100 hit a genuine `CUDA out of memory` crash -- not on the first batch, but eventually, once a
batch happened to contain several prompts that all generated close to the full 2048-token cap
simultaneously (worst-case KV-cache size for that batch). Dropping to
`--generation-batch-size 8` fixed it with no further crashes across the rest of a ~1000-row/dataset
run.

Trillium's H100s have double the VRAM (80GB vs. 40GB), so the same model/batch-size combination has
more headroom -- but the failure mode is the same one, just at a higher batch size, especially for
larger models (e.g. a ~54GB-in-bf16 27B model leaves much less slack for a worst-case KV cache than
an 8-12B model does). If a job OOMs partway through a dataset: it's safe to `scancel`, halve
`BATCH_SIZE`, and resubmit the identical command -- the checkpoint resume picks up exactly where it
left off, no rows are lost, only the crashed batch's GPU time is wasted.

## Self-resubmission

`submit_generate.sbatch` checks for `summary.json` next to `--out` after the generation script
exits; if absent, it `sbatch --dependency=afterany:$SLURM_JOB_ID`s itself with the same parameters,
tracked by a `.resubmit_count` file capped at `MAX_RESUBMITS` (default 20) so a genuinely broken run
fails loudly instead of looping forever. This is Trillium's own documented pattern in place of a
confirmed SLURM auto-`--requeue`.

`python -u` is used for the generation subprocess specifically so `tail`-ing the SLURM log while a
job is running shows real-time progress -- without it, Python fully buffers stdout when it's
redirected to a file, so the log can look stalled for a long time even while `checkpoint.jsonl`
(flushed explicitly, separately from `print()`) is genuinely being written to.

## Judging

Judge scripts (`run_social_sycophancy_judge_oeq.py`, `run_moral_sycophancy_judge_aita.py`, and the
incremental variants `incremental_judge_social.py` / `incremental_judge_moral_flip.py` for
extending an already-judged dataset without re-paying for rows already done) are pure Anthropic API
calls -- no GPU needed. Run them on the login node or locally against a synced `checkpoint.jsonl`;
never submit them as a GPU job.

## Syncing results back

```bash
cd $PROJECT/sycoscope-repo
git add -f tool_calling/tasks/sycophancy/results/generations/multimodel
git commit -m "..."
git push
```

Add `--mail-type=END,FAIL --mail-user=<email>` to `submit_generate.sbatch` for completion
notifications instead of manually polling `squeue`.

## Things to verify empirically before relying on them

- **`--gpus-per-node=h100:1` GPU-type syntax** -- if `sbatch` rejects the `h100:` prefix, drop it to
  plain `--gpus-per-node=1` in `hpc/trillium/submit_generate.sbatch` (the GPU subcluster is
  H100-only, so the type qualifier may not even be necessary).
- **`apptainer build --remote` availability/quota** for your account, if you don't have local root
  to build `sycoscope.sif` directly.
