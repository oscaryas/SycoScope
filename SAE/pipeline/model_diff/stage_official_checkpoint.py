"""Stage an authorized Hugging Face model on the GPU host without copying a token.

The local stage command uses the existing Hugging Face login or a local
HF_TOKEN entry in an explicitly named .env file to obtain
short-lived resolved file URLs and sends them to the remote download command
over SSH stdin. No login token is written on the instance.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_local_hf_token(path):
    """Load only HF_TOKEN; never execute the .env as shell code."""
    values = []
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^\s*(?:export\s+)?HF_TOKEN\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        value = match.group(1)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        values.append(value)
    if len(values) != 1 or not values[0].startswith("hf_") or any(c.isspace() for c in values[0]):
        raise ValueError("HF_TOKEN entry missing or malformed; token not shown")
    os.environ["HF_TOKEN"] = values[0]


def stage(args):
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    if args.env_file:
        load_local_hf_token(args.env_file)
    info = HfApi().model_info(args.repo, token=True)
    names = sorted(x.rfilename for x in info.siblings if
                   x.rfilename in ("config.json", "generation_config.json", "model.safetensors.index.json")
                   or x.rfilename.startswith("model-") and x.rfilename.endswith(".safetensors"))
    if args.weights_only:
        names = [name for name in names if name.endswith(".safetensors")]
    if not names or (not args.weights_only and "config.json" not in names) or not any(
            n.endswith(".safetensors") for n in names):
        raise ValueError("official checkpoint files not found")
    payload = []
    for name in names:
        meta = get_hf_file_metadata(hf_hub_url(args.repo, name), headers=build_hf_headers(token=True))
        payload.append({"name": name, "url": meta.location, "size": meta.size})
    cmd = ["ssh", "-p", str(args.port), "-i", str(args.key), "-o", "IdentitiesOnly=yes",
           "-o", "BatchMode=yes", args.host, "/venv/main/bin/python",
           args.remote_script, "download", "--dest", args.dest]
    print("staging", args.repo, "files", len(payload), "bytes", sum(x["size"] for x in payload),
          flush=True)
    result = subprocess.run(cmd, input=json.dumps(payload), text=True, check=True)
    print("remote download exit", result.returncode, flush=True)


def download(args):
    entries = json.load(sys.stdin)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    def one(entry):
        name = entry["name"]
        if Path(name).name != name:
            raise ValueError("invalid remote filename")
        target = dest / name
        expected = entry["size"]
        if target.is_file() and target.stat().st_size == expected:
            return name, "cached"
        partial = dest / (name + ".partial")
        if target.is_file() and 0 < target.stat().st_size < expected and not partial.exists():
            target.replace(partial)
        existing = partial.stat().st_size if partial.exists() else 0
        if existing >= expected:
            partial.unlink()
            existing = 0
        request = urllib.request.Request(
            entry["url"], headers={"Range": f"bytes={existing}-"} if existing else {})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("ab" if existing else "wb") as out:
            if existing and response.status != 206:
                raise ValueError(f"CDN did not honor resume range for {name}")
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if partial.stat().st_size != expected:
            raise ValueError(f"size mismatch for {name}: {partial.stat().st_size} != {expected}")
        partial.replace(target)
        return name, "downloaded"

    with ThreadPoolExecutor(max_workers=min(4, len(entries))) as pool:
        for future in as_completed(pool.submit(one, x) for x in entries):
            name, status = future.result()
            print(name, status, flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--repo", required=True)
    s.add_argument("--host", default="root@82.66.51.122")
    s.add_argument("--port", type=int, default=22933)
    s.add_argument("--key", type=Path, default=Path.home() / ".ssh/vast_vla")
    s.add_argument("--remote-script", default="/workspace/syco_pilot/stage_official_checkpoint.py")
    s.add_argument("--dest", default="/workspace/syco_pilot/base_model")
    s.add_argument("--weights-only", action="store_true",
                   help="Use CDN-signed weight URLs; stage small metadata files separately")
    s.add_argument("--env-file", type=Path,
                   help="Load only HF_TOKEN from this local .env; never transfer the token")
    d = sub.add_parser("download")
    d.add_argument("--dest", required=True)
    a = p.parse_args()
    stage(a) if a.mode == "stage" else download(a)


if __name__ == "__main__":
    main()
