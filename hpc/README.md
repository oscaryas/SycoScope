# Running SycoScope generation on SLURM (CSCS Clariden + Compute Canada Trillium)

Deployment plumbing only -- `social_generate.py` / `moral_generate.py` and the judge scripts are
unchanged. They already resume by counting lines in `--out` for a given `--seed`, which is exactly
what makes them safe to run under SLURM's wall-clock limits: a killed job can be resubmitted
identically and it picks up where it left off. Both sbatch scripts in this directory automate that
resubmission themselves (see "Self-resubmission" below).

## One-time setup (per cluster)

### 1. Build and publish the container image

```bash
docker build -t <registry>/sycoscope:latest -f hpc/Dockerfile .
docker push <registry>/sycoscope:latest
```

Replace `<registry>` with wherever you're pushing (Docker Hub, GHCR, etc.). The image does **not**
contain the repo code -- only the Python 3.13 runtime and pinned dependencies (`hpc/Dockerfile`).
Repo code is always read live from a `git clone` on the cluster, so editing
`tool_calling/tasks/sycophancy/scripts/*.py` never requires rebuilding or re-pushing the image.

### 2. Clone the repo onto persistent (non-purged) storage

- **CSCS Clariden**: `/capstor/store/<tenant>/<group>/sycoscope-repo` -- not `$SCRATCH`
  (`/iopsstor/scratch/cscs/$USER`), which is purge-swept every **14 days** and would silently
  delete in-progress checkpoints on a job that self-resubmits across weeks.
- **Trillium**: `$PROJECT/sycoscope-repo` -- not `$SCRATCH` (no fixed purge cadence yet, but
  documented as ephemeral and not meant for anything you can't afford to lose).

```bash
git clone <this-repo-url> sycoscope-repo
```

Replace `CHANGE_ME_TENANT` / `CHANGE_ME_GROUP` / `CHANGE_ME_ACCOUNT` placeholders in
`hpc/cscs/generate.edf.toml` and `hpc/cscs/submit_generate.sbatch` with your actual CSCS project
IDs, and `CHANGE_ME_ACCOUNT` in `hpc/trillium/submit_generate.sbatch` with your Alliance account
string (`rrg-...` / `rpp-...` / `def-...`). Also replace `REGISTRY_PLACEHOLDER` in
`hpc/cscs/generate.edf.toml` with the image ref you pushed in step 1.

### 3. Get the container image onto the cluster

- **CSCS Clariden**: nothing to do -- the EDF's `image =` field is a registry ref; the Container
  Engine auto-pulls and caches it the first time a job references it.
- **Trillium**: compute nodes have **no internet at all** (confirmed in Trillium's own docs), so
  the image must be built into a `.sif` off-cluster and transferred once:

  ```bash
  # off-cluster, or apptainer build --remote if you don't have local root
  apptainer build sycoscope.sif docker://<registry>/sycoscope:latest
  scp sycoscope.sif trillium:$PROJECT/sycoscope.sif
  ```

### 4. Secrets

Create `~/.sycoscope_secrets.env` on each cluster (outside the repo, never committed):

```bash
cat > ~/.sycoscope_secrets.env <<'EOF'
export HF_TOKEN=hf_...
export ANTHROPIC_API_KEY=sk-ant-...
EOF
chmod 600 ~/.sycoscope_secrets.env
```

Both sbatch scripts `source` this file. Never `cat`/`echo` it, and never commit it.

### 5. Prefetch model weights

Run once per cluster, on the **login node** (no GPU needed -- this is a plain download):

```bash
export HF_HOME=/capstor/store/<tenant>/<group>/huggingface   # CSCS
# export HF_HOME=$PROJECT/huggingface                         # Trillium
source ~/.sycoscope_secrets.env
python hpc/prefetch_weights.py
```

Generation jobs run with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` and will fail fast if a
model wasn't staged here first. This is mandatory on Trillium (no compute-node internet) and
strongly recommended on CSCS too, even though Clariden's compute nodes technically have outbound
internet via NAT -- it's a shared, rate-limit-risk egress IP that CSCS asks users not to hammer
with bulk model pulls from many concurrent jobs.

## Submitting jobs

### Smoke test first (both clusters)

Before trusting either cluster with a real `--n 2000` job, run one small job end-to-end and confirm
it produces a `checkpoint.jsonl` and (for a small enough `--n`) a `summary.json`:

```bash
sbatch --time=00:15:00 --export=ALL,MODEL="Qwen/Qwen3-8B",DATASET=ss,N=8,BATCH_SIZE=4 \
    hpc/cscs/submit_generate.sbatch        # or hpc/trillium/submit_generate.sbatch
```

Then test the resume path deliberately: `scancel` the job partway through, resubmit the identical
command, and confirm it picks up from the row count already in `checkpoint.jsonl` rather than
restarting from zero.

### Full matrix

```bash
cd sycoscope-repo   # the persistent clone from step 2
hpc/submit_matrix.sh cscs
hpc/submit_matrix.sh trillium
```

Submits all 3 models x 4 datasets (12 jobs) at the default `--n 2000`, `--generation-batch-size 16`.
Override via env vars, e.g. `N=1000 BATCH_SIZE=8 hpc/submit_matrix.sh cscs`. Each job self-resubmits
(see below) until `summary.json` appears or `MAX_RESUBMITS` (default 20) is hit.

## Self-resubmission

Neither cluster has a confirmed SLURM auto-`--requeue` we can rely on, and Trillium's own docs
explicitly recommend self-resubmission for jobs that outrun the wall-clock limit. Both
`submit_generate.sbatch` scripts do this automatically: after the generation script exits, they
check whether `summary.json` exists next to `--out`; if not, they `sbatch
--dependency=afterany:$SLURM_JOB_ID` themselves with the same parameters, incrementing a
`.resubmit_count` file capped at `MAX_RESUBMITS` (default 20) so a genuinely broken run fails loudly
instead of looping forever.

## Judging

Judge scripts (`run_social_sycophancy_judge_oeq.py`, `run_moral_sycophancy_judge_aita.py`) are pure
Anthropic API calls -- they import `torch` transitively but never load a model or touch a GPU. Run
them locally against the synced `checkpoint.jsonl` files, or on a cluster login node; never submit
them as a GPU job, since they don't need one.

## Syncing results back

`--out` and `summary.json` live inside the persistent repo clone, so:

```bash
cd sycoscope-repo
git add -f tool_calling/tasks/sycophancy/results/generations/multimodel
git commit -m "..."
git push
```

or `rsync`/`git pull` from your local machine. Add `--mail-type=END,FAIL --mail-user=<email>` to
either sbatch script for completion notifications instead of manually polling.

## Cluster facts this design assumes

| | CSCS Clariden | Compute Canada Trillium |
|---|---|---|
| GPU/node | GH200, 4/node, 96GB HBM3 | H100 SXM 80GB, 4/node |
| GPU allocation granularity | any `--gpus-per-node=N` | **1 GPU or a full node (4) only** |
| Container tooling | Container Engine + EDF (`generate.edf.toml`) | Apptainer, `.sif` built off-cluster |
| Compute-node internet | Yes, via NAT (shared IP, rate-limit risk) | **No** -- confirmed in cluster docs |
| Wall-clock (`normal`/`compute`) | 12h | 24h |
| Account flag | `--account=<CSCS project>` | `--account=<rrg-/rpp-/def->` |
| Persistent storage | `/capstor/store/<tenant>/<group>` | `$PROJECT` |

## Things to verify empirically before relying on them

- **CSCS compute-node internet without proxy vars**: `srun --environment=hpc/cscs/generate.edf.toml
  curl -sI https://huggingface.co`. Only matters as a belt-and-suspenders fallback -- the design
  above doesn't depend on it (jobs run with `HF_HUB_OFFLINE=1`).
- **`apptainer build --remote` availability/quota** for your Compute Canada account, if you don't
  have local root to build `sycoscope.sif` directly.
- **`--gpus-per-node=h100:1` GPU-type syntax on Trillium** -- if `sbatch` rejects the `h100:` prefix,
  drop it to plain `--gpus-per-node=1` in `hpc/trillium/submit_generate.sbatch` (Trillium's GPU
  subcluster is H100-only, so the type qualifier may not even be necessary).
- **Plain `sbatch --array` support on CSCS** -- not load-bearing for this design (12 jobs are
  submitted individually via `submit_matrix.sh`, not as an array), but worth knowing if the job
  count grows later.
