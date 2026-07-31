import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # tools.py -> sae -> tasks -> tool_calling -> repo_root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SAE.sae import train as sae_train
from SAE.sae.data import DEFAULT_CACHE_DIR, load_cache_meta

_OUTPUT_ROOT = REPO_ROOT / "SAE" / "results" / "runs"


def set_output_dir(path: Path):
    global _OUTPUT_ROOT
    _OUTPUT_ROOT = Path(path)


def list_activation_cache() -> dict:
    """
    Report which layers are cached, their hidden_size, and available
    granularities, so train_sae's inputs don't have to be guessed.
    """
    try:
        meta = load_cache_meta(DEFAULT_CACHE_DIR)
    except FileNotFoundError:
        return {"error": f"no activation cache at {DEFAULT_CACHE_DIR} -- run SAE/pipeline/cache_activations.py first"}
    return {
        "layers": meta["layers"],
        "hidden_size": meta["hidden_size"],
        "granularities": ["sentence", "response"],
        "n_sentences": meta.get("n_sentences"),
        "n_responses": meta.get("n_responses"),
    }


def train_sae(layer: int, granularity: str, dict_size: int, k: int, steps: int = 2000) -> str:
    """
    Train one TopK sparse autoencoder on the given layer/granularity with the
    given dictionary size and top-k sparsity. Resumes automatically from a
    checkpoint if a run with this exact (layer, granularity, dict_size, k)
    already exists and hasn't reached `steps` yet -- call again with a larger
    `steps` to keep training a promising config further.
    """
    config = {
        "layer": layer,
        "granularity": granularity,
        "dict_size": dict_size,
        "k": k,
        "steps": steps,
        "k_aux": 256,
        "aux_coef": 1.0 / 32,
        "dead_steps": 200,
        "lr": 3e-4,
        "batch_size": 256,
        "eval_every": 200,
        "cache_dir": str(DEFAULT_CACHE_DIR),
        "output_root": str(_OUTPUT_ROOT),
        "seed": 0,
    }
    metrics = sae_train.run_training(config)
    return (
        f"run_id={metrics['run_id']} step={metrics['step']} "
        f"fvu={metrics['fvu']:.4f} l0={metrics['l0']:.2f} dead_frac={metrics['dead_frac']:.4f}"
    )


def read_sweep_log() -> str:
    """
    Return the last ~20 runs' configs and final metrics (one JSON object per
    line), so prior configs and their results don't need to be rerun to see.
    """
    path = _OUTPUT_ROOT / "sweep_log.jsonl"
    if not path.exists():
        return "no runs yet"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return "\n".join(lines[-20:])


def write_analysis(text: str, filename: str = "sae_analysis.md") -> str:
    """Write text to filename under the current output directory. Default filename is sae_analysis.md."""
    out = _OUTPUT_ROOT / filename
    out.write_text(text, encoding="utf-8")
    return f"{filename} written ({len(text)} chars)"


TOOLS = {
    "list_activation_cache": list_activation_cache,
    "train_sae": train_sae,
    "read_sweep_log": read_sweep_log,
    "write_analysis": write_analysis,
}
