#!/usr/bin/env python3
"""Finish an already-started, user-authorized Vast SYCON run and download results.

No instances are created/stopped, no generation is launched, and no remote files
are deleted. Requires the prepared remote /workspace/syconbench_eval workspace.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
REMOTE = "/workspace/syconbench_eval"
RUN_NAME = "llama31_5k_subset"
RUN = ROOT / "prompt_probes/results" / RUN_NAME
FILES = ("activations.npz", "activations_index.jsonl", "meta.json", "preparation.json", "records.jsonl")


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def supervisor_status(ssh, name):
    """Supervisor uses exit 3 for valid non-running states, not just errors."""
    result = subprocess.run(ssh + [f"supervisorctl status {shlex.quote(name)}"],
                            text=True, stdout=subprocess.PIPE)
    fields = result.stdout.strip().split()
    states = {"STOPPED", "STARTING", "RUNNING", "BACKOFF", "STOPPING", "EXITED", "FATAL", "UNKNOWN"}
    if result.returncode not in (0, 3):
        result.check_returncode()
    if len(fields) < 2 or fields[0] != name or fields[1] not in states:
        raise RuntimeError(f"invalid supervisor status: {result.stdout.strip()}")
    return result.stdout.strip(), fields[1]


def validate_saved_judgments(generations, path):
    judged = [json.loads(line) for line in path.read_text().splitlines()]
    by_id = {row["id"]: row for row in judged}
    if len(judged) != len(generations) or len(by_id) != len(judged):
        raise ValueError("saved judgments do not cover the generation set uniquely")
    for row in generations:
        saved = by_id.get(row["id"], {})
        for field in ("messages", "model", "question", "setting", "turn_finish_reasons"):
            if saved.get(field) != row.get(field):
                raise ValueError(f"saved judgment input differs: {row['id']} / {field}")
        labels = saved.get("turn_judgments", [])
        if len(labels) != 5:
            raise ValueError(f"missing turn judgments: {row['id']}")
        for turn, reason in enumerate(row["turn_finish_reasons"]):
            if reason != "stop":
                break  # Truncated turns and their subsequent histories are excluded.
            if labels[turn] not in (0, 1):
                raise ValueError(f"unjudged eligible turn: {row['id']} / {turn + 1}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--resume-from-judgments", action="store_true",
                        help="Validate local completed judgments; skip generation download and all judge calls.")
    args = parser.parse_args()
    ssh = ["ssh", "-i", str(args.identity), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=15", "-p", args.port, args.host]
    scp = ["scp", "-i", str(args.identity), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-P", args.port]
    work = RUN / "syconbench_run"
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / "workflow_status.json"

    def status(stage, **extra):
        data = {"stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
        report_path.write_text(json.dumps(data, indent=2))
        print(json.dumps(data), flush=True)

    def remote(command):
        return subprocess.run(ssh + [command], check=True, text=True, stdout=subprocess.PIPE).stdout.strip()

    def wait_job(name, logfile):
        while True:
            try:
                result, state = supervisor_status(ssh, name)
                log = remote(f"tail -n 2 {shlex.quote(logfile)}")
            except subprocess.CalledProcessError as exc:
                if exc.returncode != 255:
                    raise
                status("connection_retry", job=name)
                time.sleep(30)
                continue
            print(result, flush=True)
            print(log, flush=True)
            if state == "EXITED":
                return
            if state not in ("RUNNING", "STARTING"):
                raise RuntimeError(f"job needs attention: {result}")
            time.sleep(45)

    def local(*argv, stdout=None):
        subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT, check=True, stdout=stdout, stderr=stdout)

    def download(remote_path, local_path):
        subprocess.run(scp + [f"{args.host}:{remote_path}", str(local_path)], check=True)

    def upload(local_path, remote_path):
        subprocess.run(scp + [str(local_path), f"{args.host}:{remote_path}"], check=True)

    if not args.resume_from_judgments:
        status("waiting_for_generation")
        wait_job("syconbench_generation", REMOTE + "/results/generation.log")
        for filename in ("generations.jsonl", "generations.meta.json"):
            download(REMOTE + "/results/" + filename, work / filename)
    generations = [json.loads(l) for l in (work / "generations.jsonl").read_text().splitlines()]
    if len(generations) != 500 or len({r["id"] for r in generations}) != 500:
        raise ValueError("generation did not complete all 500 unique scenarios")
    if args.resume_from_judgments:
        validate_saved_judgments(generations, work / "judge/judged.jsonl")
        status("resuming_saved_judgments", n_conversations=len(generations))
    else:
        status("judging", n_conversations=len(generations))
        local("-u", ROOT / "probing/evaluations/prompt_probes/judge/judge_syconbench_budgeted.py",
              "--input", work / "generations.jsonl", "--output-dir", work / "judge",
              "--env-file", ROOT / ".env", "--budget-usd", "49.75", "--workers", "12")
    upload(work / "judge/judged.jsonl", REMOTE + "/results/judged.jsonl")
    upload(ROOT / "probing/analyze_probes/syconbench_extraction.supervisor.conf",
           "/etc/supervisor/conf.d/syconbench_extraction.conf")
    remote("supervisorctl reread && supervisorctl update")
    remote_cache = REMOTE + "/prompt_probes/results/" + RUN_NAME + "/eval_syconbench"
    cache_exists = remote(f"test -f {remote_cache}/activations.npz && echo yes || echo no") == "yes"
    if not cache_exists:
        status("extracting_activations")
        current, state = supervisor_status(ssh, "syconbench_extraction")
        if state == "STOPPED":
            remote("supervisorctl start syconbench_extraction")
        elif state not in ("RUNNING", "STARTING"):
            raise RuntimeError(f"extraction needs attention before restart: {current}")
        wait_job("syconbench_extraction", REMOTE + "/results/extraction.log")
    status("downloading_activations")
    dest = RUN / "eval_syconbench"
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for filename in FILES:
        expected = remote(f"sha256sum {remote_cache}/{filename}").split()[0]
        path = dest / filename
        if not path.exists() or checksum(path) != expected:
            part = dest / (filename + ".download")
            download(remote_cache + "/" + filename, part)
            if checksum(part) != expected:
                raise ValueError(f"download checksum mismatch: {filename}")
            if path.exists():
                raise ValueError(f"refusing to replace different existing artifact: {path}")
            part.rename(path)
        hashes[filename] = expected
    import numpy as np
    index = [json.loads(l) for l in (dest / "activations_index.jsonl").read_text().splitlines()]
    meta = json.loads((dest / "meta.json").read_text())
    if meta["model"] != "meta-llama/Llama-3.1-8B-Instruct":
        raise ValueError("downloaded model identity differs")
    with np.load(dest / "activations.npz") as arrays:
        if len(arrays.files) != 24:
            raise ValueError("expected all eight layers and three positions")
        for key in arrays.files:
            if arrays[key].shape != (len(index), 4096) or not np.isfinite(arrays[key]).all():
                raise ValueError(f"invalid activation array: {key}")
    (work / "download_checksums.json").write_text(json.dumps(hashes, indent=2))
    status("scoring_probes", n_turns=len(index))
    os.environ.update(MPLCONFIGDIR="/private/tmp/sycoscope-sycon-matplotlib",
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    with (work / "evaluation.log").open("w") as log:
        local("-u", ROOT / "probing/analyze_probes/eval_syconbench.py", "--run-name", RUN_NAME,
              "--judged", work / "judge/judged.jsonl", stdout=log)
        local("-u", ROOT / "probing/analyze_probes/eval_matrix.py", "--run-name", RUN_NAME,
              "--target", "syconbench", "--n-boot", "1000", stdout=log)
        local("-u", ROOT / "probing/analyze_probes/summarize_syconbench.py", "--run-name", RUN_NAME, stdout=log)
    status("complete", report=str(RUN / "analysis/syconbench/RESULTS.md"))


if __name__ == "__main__":
    main()
