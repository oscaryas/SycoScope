import json
import sys
from pathlib import Path

_utils = Path(__file__).resolve().parent.parent
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


def generate_moral_sycophancy_labels(n_pairs: int = 50, output_path: str = "behavioral_labels.jsonl") -> str:
    """
    Judge n_pairs AITA-NTA-FLIP conflicts (original_post vs flipped_story,
    same underlying conflict) with an LLM judge for a YTA/NTA verdict, then
    label every response 1 (moral sycophancy) if both sides of its conflict
    were told NTA, else 0. Writes output_path in the same {"text","label"}
    format generate_behavioral_labels uses, so extract_activations /
    train_probe_family / write_metrics / steer_and_generate all work
    unchanged on the result. Requires ANTHROPIC_API_KEY. Skips if
    output_path already exists with valid text+label records.
    """
    out = _OUTPUT_DIR / output_path
    if _is_valid_labels_file(out):
        n = sum(1 for l in out.read_text().splitlines() if l.strip())
        return f"Skipping — {out} already exists with {n} valid examples"
    if _session["tokenizer"] is None:
        return "error: no model loaded — call load_model first (its tokenizer builds the chat-formatted text)"

    import moral_sycophancy_judge

    result = moral_sycophancy_judge.generate_moral_sycophancy_labels(_session["tokenizer"], n_pairs=n_pairs)
    with open(out, "w", encoding="utf-8") as f:
        for rec in result["records"]:
            f.write(json.dumps(rec) + "\n")

    return (
        f"Judged {result['n_pairs_judged']} conflicts ({result['n_skipped_other']} skipped, unclear verdict). "
        f"Wrote {len(result['records'])} labeled examples to {out}. "
        f"Moral sycophancy rate (both NTA): {result['moral_sycophancy_rate']:.2%} "
        f"(both_NTA={result['n_both_nta']}, both_YTA={result['n_both_yta']}, mixed={result['n_mixed']})"
    )


def generate_social_sycophancy_labels(
    metric: str,
    dataset: str = "OEQ",
    n_examples: int = 50,
    output_path: str = "behavioral_labels.jsonl",
) -> str:
    """
    Judge n_examples responses from SAE/results/{dataset}.jsonl (e.g. "OEQ" or
    "SS") independently for one social-sycophancy metric, adapted from the
    ELEPHANT paper's sycophancy_scorers.py:
    - "validation": is the response emotionally validating?
    - "indirectness": does the response use suggestive rather than direct language?
    - "framing": does the response NOT challenge the question's premise (1) vs. challenge it (0)?
    Unlike generate_moral_sycophancy_labels, each response is scored independently --
    no paired counterpart needed. Writes output_path in the same {"text","label"}
    format generate_behavioral_labels uses, so extract_activations /
    train_probe_family / write_metrics / steer_and_generate all work
    unchanged. Requires ANTHROPIC_API_KEY. Skips if output_path already
    exists with valid text+label records.
    """
    if metric not in ("validation", "indirectness", "framing"):
        return f"error: metric must be 'validation', 'indirectness', or 'framing' — got '{metric}'"

    out = _OUTPUT_DIR / output_path
    if _is_valid_labels_file(out):
        n = sum(1 for l in out.read_text().splitlines() if l.strip())
        return f"Skipping — {out} already exists with {n} valid examples"
    if _session["tokenizer"] is None:
        return "error: no model loaded — call load_model first (its tokenizer builds the chat-formatted text)"

    import social_sycophancy_judge

    input_path = social_sycophancy_judge.DEFAULT_RESULTS_DIR / f"{dataset}.jsonl"
    if not input_path.exists():
        return f"error: {input_path} not found"

    result = social_sycophancy_judge.generate_social_sycophancy_labels(
        _session["tokenizer"], metric, n_examples=n_examples, input_path=input_path
    )
    with open(out, "w", encoding="utf-8") as f:
        for rec in result["records"]:
            f.write(json.dumps(rec) + "\n")

    return (
        f"Judged {result['n_judged']} {dataset} responses for '{metric}' "
        f"({result['n_skipped_error']} skipped, judge output didn't parse). "
        f"Wrote {len(result['records'])} labeled examples to {out}. "
        f"Rate (label=1): {result['rate']:.2%}"
    )


def extract_activations(labels_path: str, answer_token_id: int, pooling: str = "mean") -> str:
    """
    Extract and cache MHA, MLP, and residual activations. Call inspect_model first.
    pooling: "mean" (default) averages each activation over the response token
    span (everything after the answer_token_id delimiter) instead of reading
    just the single delimiter position -- less sensitive to exactly where
    that one token lands. "last" uses only that single position instead.
    Writes activations/ with metadata.json, labels.npy, mha.npy, mlp.npy, residual.npy.
    """
    import numpy as np
    import sycophancy_probes

    if _session["model"] is None:
        return "error: no model loaded — call load_model first"
    if _session["model_config"] is None:
        return "error: architecture unknown — call inspect_model first"
    if pooling not in ("mean", "last"):
        return f"error: pooling must be 'mean' or 'last' — got '{pooling}'"

    labels_file = _OUTPUT_DIR / labels_path
    if not labels_file.exists():
        return f"error: labels file not found at {labels_file}"

    records = [json.loads(l) for l in labels_file.read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]

    config = {**_session["model_config"], "answer_token_id": answer_token_id}
    activations = sycophancy_probes.collect_activations(
        _session["model"], _session["tokenizer"], texts, config, batch_size=1, pooling=pooling
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
        "pooling": pooling,
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
        acc, ci, states, auc, auc_ci = sycophancy_probes.train_mha_probes(acts, labels, n_layers, mc["n_heads"])
        _session["probe_results"]["mha"] = {"accuracy": acc, "ci": ci, "states": states, "auc": auc, "auc_ci": auc_ci}
        best = max(acc.values()) if acc else 0.0
        return f"MHA probes done. Best: {best:.3f} ({n_layers * mc['n_heads']} probes)"
    elif probe_type == "mlp":
        acts = np.load(cache_dir / "mlp.npy")
        acc, ci, states, auc, auc_ci = sycophancy_probes.train_mlp_probes(acts, labels, n_layers)
        _session["probe_results"]["mlp"] = {"accuracy": acc, "ci": ci, "states": states, "auc": auc, "auc_ci": auc_ci}
        best = max(acc.values()) if acc else 0.0
        return f"MLP probes done. Best: {best:.3f} ({n_layers} probes)"
    else:
        acts = np.load(cache_dir / "residual.npy")
        acc, ci, states, auc, auc_ci = sycophancy_probes.train_residual_probes(acts, labels, n_layers)
        _session["probe_results"]["residual"] = {"accuracy": acc, "ci": ci, "states": states, "auc": auc, "auc_ci": auc_ci}
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
        "mha_auc": pr["mha"].get("auc", {}),
        "mha_auc_ci": pr["mha"].get("auc_ci", {}),
        "mlp_accuracy": pr["mlp"]["accuracy"],
        "mlp_ci": pr["mlp"]["ci"],
        "mlp_states": pr["mlp"].get("states", {}),
        "mlp_auc": pr["mlp"].get("auc", {}),
        "mlp_auc_ci": pr["mlp"].get("auc_ci", {}),
        "residual_accuracy": pr["residual"]["accuracy"],
        "residual_ci": pr["residual"]["ci"],
        "residual_states": pr["residual"].get("states", {}),
        "residual_auc": pr["residual"].get("auc", {}),
        "residual_auc_ci": pr["residual"].get("auc_ci", {}),
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


def list_steering_vectors(probe_dir: str = "final_probe") -> dict:
    """
    Report which (component, key) steering vectors are available in probe_dir
    (written by write_metrics). A component only appears once
    train_probe_family has been called for it and write_metrics has run.
    """
    from sycophancy_steering import load_steering_vectors

    out_dir = _OUTPUT_DIR / probe_dir
    available = {}
    for component in ("mha", "mlp", "residual"):
        try:
            vectors = load_steering_vectors(str(out_dir), component)
            available[component] = sorted(str(k) for k in vectors.keys())
        except FileNotFoundError:
            available[component] = []
    return available


def steer_and_generate(
    component: str,
    layer: int,
    alpha: float,
    prompt: str,
    head: int = 0,
    max_new_tokens: int = 150,
    probe_dir: str = "final_probe",
) -> dict:
    """
    Add alpha * (probe direction, scaled to ~1 projection std) to component
    "mha" (uses head), "mlp", or "residual" at the given layer, then
    generate from prompt. Returns both the unsteered baseline and the
    steered continuation so they can be compared directly. Call
    list_steering_vectors first to see which (component, layer[, head])
    combinations are available.
    """
    if _session["model"] is None:
        return "error: no model loaded — call load_model first"
    if _session["model_config"] is None:
        return "error: architecture unknown — call inspect_model first"

    from sycophancy_steering import ActivationSteerer, load_steering_vectors

    out_dir = _OUTPUT_DIR / probe_dir
    vectors = load_steering_vectors(str(out_dir), component)
    key = (layer, head) if component == "mha" else layer
    if key not in vectors:
        return f"error: no steering vector for component={component} key={key}. Available: {sorted(str(k) for k in vectors)}"

    steerer = ActivationSteerer(_session["model"], _session["tokenizer"], _session["model_config"])
    baseline = steerer.generate(prompt, max_new_tokens)
    steerer.attach(component, layer, vectors[key], alpha, head=head if component == "mha" else None)
    steered = steerer.generate(prompt, max_new_tokens)
    steerer.cleanup()

    return {"baseline": baseline, "steered": steered}


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
    "generate_moral_sycophancy_labels": generate_moral_sycophancy_labels,
    "generate_social_sycophancy_labels": generate_social_sycophancy_labels,
    "extract_activations": extract_activations,
    "train_probe_family": train_probe_family,
    "write_metrics": write_metrics,
    "fetch_paper_results": fetch_paper_results,
    "compare_with_paper": compare_with_paper,
    "list_steering_vectors": list_steering_vectors,
    "steer_and_generate": steer_and_generate,
    "write_analysis": write_analysis,
}
