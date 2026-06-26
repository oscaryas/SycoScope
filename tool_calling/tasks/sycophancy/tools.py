import json
import sys
from pathlib import Path

_utils = Path(__file__).resolve().parents[4] / "src" / "utils"
if str(_utils) not in sys.path:
    sys.path.insert(0, str(_utils))

_OUTPUT_DIR = Path(".")

_session = {
    "model": None,
    "tokenizer": None,
    "dtype": None,
    "model_path": None,
    "model_config": None,
    "activations": None,
    "probe_results": {},
}


def set_output_dir(path: Path):
    global _OUTPUT_DIR
    _OUTPUT_DIR = Path(path)


def load_model(model_path: str) -> str:
    """Load model and tokenizer with conservative bfloat16 policy, device_map=auto."""
    import gpu_memory
    model, tokenizer, dtype = gpu_memory.load_model_conservatively(model_path)
    _session.update({"model": model, "tokenizer": tokenizer, "dtype": dtype, "model_path": model_path})
    return f"Model '{model_path}' loaded (dtype={dtype})"


def cleanup_model() -> str:
    """Free GPU memory. Call between models in cross-model analysis."""
    if _session["model"] is None:
        return "error: no model loaded — call load_model first"
    import gpu_memory
    gpu_memory.safe_cleanup(model=_session["model"], tokenizer=_session["tokenizer"])
    _session.update({"model": None, "tokenizer": None, "dtype": None,
                     "model_path": None, "model_config": None, "activations": None,
                     "probe_results": {}})
    return "Model cleaned up, GPU memory released"


def inspect_model() -> dict:
    """
    Auto-discover architecture: n_layers, n_heads, hidden_dim, head_dim, mlp_dim,
    mha/mlp hook patterns and full hook paths. head_dim read from actual weight shape
    to handle grouped-query attention. Fails closed if hooks are missing.
    """
    if _session["model"] is None:
        return "error: no model loaded — call load_model first"

    model = _session["model"]
    mha_hook_paths, mlp_hook_paths = [], []
    hidden_dim = head_input_dim = mlp_dim = None

    for name, module in model.named_modules():
        if name.endswith("self_attn.o_proj"):
            mha_hook_paths.append(name)
            if hidden_dim is None:
                hidden_dim = module.out_features
                head_input_dim = module.in_features
        if name.endswith("mlp.down_proj"):
            mlp_hook_paths.append(name)
            if mlp_dim is None:
                mlp_dim = module.in_features

    if not mha_hook_paths:
        raise RuntimeError("inspect_model: no 'self_attn.o_proj' modules found")
    if not mlp_hook_paths:
        raise RuntimeError("inspect_model: no 'mlp.down_proj' modules found")

    n_layers = len(mha_hook_paths)
    if len(mlp_hook_paths) != n_layers:
        raise RuntimeError(
            f"inspect_model: MHA hooks ({n_layers}) != MLP hooks ({len(mlp_hook_paths)})"
        )

    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    n_heads = getattr(cfg, "num_attention_heads", None)
    if n_heads is None:
        raise RuntimeError("inspect_model: cannot read num_attention_heads from model.config")

    config = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "hidden_dim": hidden_dim,
        "head_dim": head_input_dim // n_heads,
        "mlp_dim": mlp_dim,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "mha_hook_paths": mha_hook_paths,
        "mlp_hook_paths": mlp_hook_paths,
        "residual_available": True,
    }
    _session["model_config"] = config
    return config


def get_answer_token_id(sample_prompt: str) -> int | str:
    """
    Return the delimiter token ID used by the existing activation extractor to
    choose an answer position. Despite the name, this is not always the first
    generated answer token. Checks for <end_of_turn> (Gemma), <|eot_id|>
    (Llama), <|im_end|> (Qwen) in that order. Falls back to eos_token_id.
    """
    if _session["tokenizer"] is None:
        return "error: no model loaded — call load_model first"
    tokenizer = _session["tokenizer"]
    unk_id = getattr(tokenizer, "unk_token_id", -1)
    for token_str in ["<end_of_turn>", "<|eot_id|>", "<|im_end|>"]:
        encoded = tokenizer.encode(token_str, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] != unk_id:
            return encoded[0]
    return tokenizer.eos_token_id


def _is_valid_labels_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            return False
        labels = [r.get("label") for r in lines]
        if not all("text" in r and r.get("label") in (0, 1) for r in lines):
            return False
        return len(set(labels)) >= 2
    except Exception:
        return False


def generate_behavioral_labels(n_examples: int, output_path: str = "behavioral_labels.jsonl") -> str:
    """
    Generate two-turn sycophancy dataset from TruthfulQA using the loaded model.
    Skips if output_path already exists with valid text+label records.
    """
    out = _OUTPUT_DIR / output_path
    if _is_valid_labels_file(out):
        n = sum(1 for l in out.read_text().splitlines() if l.strip())
        return f"Skipping — {out} already exists with {n} valid examples"
    if _session["model"] is None:
        return "error: no model loaded — call load_model first"

    import sycophancy_data
    items = sycophancy_data.load_truthfulqa(n_examples)
    all_examples = []
    with open(out, "w") as f:
        for idx, item in enumerate(items):
            examples = sycophancy_data.build_examples(
                _session["model"], _session["tokenizer"], item, idx
            )
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
            all_examples.extend(examples)

    rate = sycophancy_data.compute_sycophancy_rate(all_examples)
    return (
        f"Generated {len(all_examples)} examples to {out}. "
        f"Sycophancy rate: {rate:.3f}. "
        f"Label=1: {sum(e['label']==1 for e in all_examples)}, "
        f"Label=0: {sum(e['label']==0 for e in all_examples)}"
    )


def extract_activations(labels_path: str, answer_token_id: int) -> str:
    """
    Extract and cache MHA, MLP, and residual activations. Call inspect_model first.
    Writes activations/ with metadata.json, labels.npy, mha.npy, mlp.npy, residual.npy.
    """
    import numpy as np
    import sycophancy_probes

    if _session["model"] is None:
        return "error: no model loaded — call load_model first"
    if _session["model_config"] is None:
        return "error: architecture unknown — call inspect_model first"

    labels_file = _OUTPUT_DIR / labels_path
    if not labels_file.exists():
        return f"error: labels file not found at {labels_file}"

    records = [json.loads(l) for l in labels_file.read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]

    config = {**_session["model_config"], "answer_token_id": answer_token_id}
    activations = sycophancy_probes.collect_activations(
        _session["model"], _session["tokenizer"], texts, config, batch_size=1
    )

    cache_dir = _OUTPUT_DIR / "activations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "labels.npy", np.array(labels, dtype=np.int64))
    np.save(cache_dir / "mha.npy", activations["mha"].astype(np.float32))
    np.save(cache_dir / "mlp.npy", activations["mlp"].astype(np.float32))
    np.save(cache_dir / "residual.npy", activations["residual"].astype(np.float32))

    mc = _session["model_config"]
    metadata = {
        "model_name": _session["model_path"],
        "labels_path": labels_path,
        "n_examples": len(texts),
        "dtype": "float32",
        "position_strategy": "answer_token_id",
        "answer_token_id": answer_token_id,
        "model_config": {k: mc[k] for k in ("n_layers", "n_heads", "hidden_dim", "head_dim", "mlp_dim", "mha_hook", "mlp_hook")},
        "files": {"labels": "labels.npy", "mha": "mha.npy", "mlp": "mlp.npy", "residual": "residual.npy"},
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    _session["activations"] = str(cache_dir)
    return (
        f"Activations cached to {cache_dir} ({len(texts)} examples). "
        f"Shapes: mha={activations['mha'].shape}, mlp={activations['mlp'].shape}"
    )


def train_probe_family(probe_type: str) -> str:
    """
    Train probes from cached activations. probe_type: mha, mlp, or residual.
    Call extract_activations first. Call all three before write_metrics.
    """
    if _session["activations"] is None:
        return "error: no activation cache — call extract_activations first"
    if probe_type not in ("mha", "mlp", "residual"):
        return f"error: probe_type must be mha, mlp, or residual — got '{probe_type}'"

    import numpy as np
    import sycophancy_probes

    cache_dir = Path(_session["activations"])
    meta = json.loads((cache_dir / "metadata.json").read_text())
    mc = meta["model_config"]
    n_layers = mc["n_layers"]
    labels = np.load(cache_dir / "labels.npy").astype(np.float32)

    if probe_type == "mha":
        acts = np.load(cache_dir / "mha.npy")
        acc, ci, states = sycophancy_probes.train_mha_probes(acts, labels, n_layers, mc["n_heads"])
        _session["probe_results"]["mha"] = {"accuracy": acc, "ci": ci, "states": states}
        best = max(acc.values()) if acc else 0.0
        return f"MHA probes done. Best: {best:.3f} ({n_layers * mc['n_heads']} probes)"
    elif probe_type == "mlp":
        acts = np.load(cache_dir / "mlp.npy")
        acc, ci = sycophancy_probes.train_mlp_probes(acts, labels, n_layers)
        _session["probe_results"]["mlp"] = {"accuracy": acc, "ci": ci}
        best = max(acc.values()) if acc else 0.0
        return f"MLP probes done. Best: {best:.3f} ({n_layers} probes)"
    else:
        acts = np.load(cache_dir / "residual.npy")
        acc, ci = sycophancy_probes.train_residual_probes(acts, labels, n_layers)
        _session["probe_results"]["residual"] = {"accuracy": acc, "ci": ci}
        best = max(acc.values()) if acc else 0.0
        return f"Residual probes done. Best: {best:.3f} ({n_layers} probes)"


def write_metrics() -> str:
    """
    Save probe results to final_probe/ and write job-local metrics.json.
    Requires all three probe families. run_task.sh still runs evaluate.py after
    agent exit and that evaluator output is the benchmark authority.
    """
    missing = [t for t in ("mha", "mlp", "residual") if t not in _session["probe_results"]]
    if missing:
        return f"error: missing probe families: {missing}. Call train_probe_family for each."

    import sycophancy_probes

    pr = _session["probe_results"]
    results = {
        "mha_accuracy": pr["mha"]["accuracy"],
        "mha_ci": pr["mha"]["ci"],
        "mha_states": pr["mha"].get("states", {}),
        "mlp_accuracy": pr["mlp"]["accuracy"],
        "mlp_ci": pr["mlp"]["ci"],
        "residual_accuracy": pr["residual"]["accuracy"],
        "residual_ci": pr["residual"]["ci"],
    }
    metadata = sycophancy_probes.save_probe_results(
        results, str(_OUTPUT_DIR / "final_probe"), model_name=_session["model_path"] or ""
    )
    (_OUTPUT_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))
    return (
        f"metrics.json written. MHA best: {metadata['mha_best_accuracy']:.3f}, "
        f"MLP best: {metadata['mlp_best_accuracy']:.3f}, "
        f"Residual best: {metadata['residual_best_accuracy']:.3f}"
    )


def fetch_paper_results(paper_title: str) -> str:
    """
    Get paper-reported accuracies. Checks task_context/paper_results.json first
    (preferred for reproducibility). Falls back to checking _OUTPUT_DIR, then web search instructions.
    Writes paper_results.json in the flat format expected by sycophancy_compare:
    {model_name: {mha_best_accuracy, mlp_best_accuracy, residual_best_accuracy}}.
    """
    cached = _OUTPUT_DIR / "task_context" / "paper_results.json"
    if cached.exists():
        try:
            data = json.loads(cached.read_text())
            (_OUTPUT_DIR / "paper_results.json").write_text(json.dumps(data, indent=2))
            return f"Loaded from task_context/paper_results.json. Models: {list(data.keys())}"
        except Exception as e:
            return f"error parsing task_context/paper_results.json: {e}"

    # Check if already written to output dir
    output_cached = _OUTPUT_DIR / "paper_results.json"
    if output_cached.exists():
        try:
            data = json.loads(output_cached.read_text())
            return f"Found paper_results.json already in output dir. Models: {list(data.keys())}"
        except Exception as e:
            return f"error parsing paper_results.json: {e}"

    return (
        f"No paper_results.json found. Search for '{paper_title}' on arXiv. "
        "Find best MHA, MLP, residual accuracies for the models you tested. "
        "Write paper_results.json using write_analysis() with JSON: "
        '{"<model_name>": {"mha_best_accuracy": 0.X, "mlp_best_accuracy": 0.X, "residual_best_accuracy": 0.X}, ...}. '
        "Call: write_analysis(json_text, 'paper_results.json')"
    )


def compare_with_paper(results_path: str, paper_results_path: str) -> str:
    """Compare reproduced metrics against paper. Writes comparison_table.md."""
    import sycophancy_compare

    results_file = _OUTPUT_DIR / results_path
    paper_file = _OUTPUT_DIR / paper_results_path
    if not results_file.exists():
        return f"error: {results_file} not found — call write_metrics first"
    if not paper_file.exists():
        return f"error: {paper_file} not found — call fetch_paper_results first"

    results = json.loads(results_file.read_text())
    paper = json.loads(paper_file.read_text())
    model_name = results.get("model_name", _session.get("model_path") or "")
    rows = sycophancy_compare.compare(results, paper, model_name)
    table = sycophancy_compare.render_table(rows)

    (_OUTPUT_DIR / "comparison_table.md").write_text(table + "\n")
    return f"comparison_table.md written ({len(rows)} rows)"


def write_analysis(text: str, filename: str = "analysis.md") -> str:
    """Write text to filename in the output directory. Default filename is analysis.md."""
    out = _OUTPUT_DIR / filename
    out.write_text(text)
    return f"{filename} written ({len(text)} chars)"


TOOLS = {
    "load_model": load_model,
    "cleanup_model": cleanup_model,
    "inspect_model": inspect_model,
    "get_answer_token_id": get_answer_token_id,
    "generate_behavioral_labels": generate_behavioral_labels,
    "extract_activations": extract_activations,
    "train_probe_family": train_probe_family,
    "write_metrics": write_metrics,
    "fetch_paper_results": fetch_paper_results,
    "compare_with_paper": compare_with_paper,
    "write_analysis": write_analysis,
}
