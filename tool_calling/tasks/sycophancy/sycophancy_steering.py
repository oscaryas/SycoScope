"""
Activation steering for sycophancy probing: add a probe-trained direction
vector to a chosen layer's MHA head, MLP output, or residual stream during
generation, so a discovered sycophancy direction's effect can be observed
directly rather than only inferred from probe accuracy.

Reads direction vectors from the *_probe_weights.pth / *_projection_stds.pt
checkpoints that sycophancy_probes.save_probe_results already writes, and
locates modules the same way sycophancy_model_registry.register_hooks does
(hook-path suffix + layer index parsed from the module's dotted name) rather
than a separate layer_path config, so no new registry fields are needed.
"""

from pathlib import Path

import torch

from sycophancy_model_registry import _extract_layer_idx


def load_steering_vectors(probe_dir: str, component: str) -> dict:
    """
    Load direction vectors for one component ("mha", "mlp", or "residual")
    from probe_dir (as written by write_metrics -> save_probe_results).

    Returns {key: torch.Tensor}, where key is (layer, head) for "mha" or
    layer (int) for "mlp"/"residual". Each vector is a unit direction scaled
    by that probe's projection std, so alpha=1.0 means "shift by ~1 std of
    this direction's activation spread".
    """
    probe_path = Path(probe_dir)
    weights_path = probe_path / f"{component}_probe_weights.pth"
    stds_path = probe_path / f"{component}_projection_stds.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No {weights_path.name} in {probe_path} -- train_probe_family('{component}') "
            "and write_metrics must run first."
        )

    weights_ckpt = torch.load(weights_path, map_location="cpu")
    stds_ckpt = torch.load(stds_path, map_location="cpu") if stds_path.exists() else {}

    vectors = {}
    for key, state_dict in weights_ckpt.items():
        w = state_dict["linear.weight"][0]
        direction = w / (w.norm() + 1e-8)
        proj_std = stds_ckpt.get(key, 1.0)
        vectors[key] = direction * proj_std
    return vectors


def load_direction_vectors(direction_dir: str, component: str, fmt: str = "probe") -> dict:
    """
    Format-agnostic direction loader. fmt="probe" delegates to load_steering_vectors
    above. fmt="dim" reads {component}_dim_vectors.pt directly -- DIM directions are
    saved already alpha-ready (direction*proj_std), so no further scaling is needed.
    Both return the same shape: {key: torch.Tensor}, key=(layer,head) for "mha" or
    layer (int) for "mlp"/"residual" -- ActivationSteerer.attach and every
    cross-dataset sweep function are format-agnostic once handed this dict, so this
    is the only place that needs to know DIM and probe checkpoints differ.
    """
    if fmt == "probe":
        return load_steering_vectors(direction_dir, component)
    if fmt == "dim":
        vectors_path = Path(direction_dir) / f"{component}_dim_vectors.pt"
        if not vectors_path.exists():
            raise FileNotFoundError(f"No {vectors_path.name} in {direction_dir}")
        return torch.load(vectors_path, map_location="cpu")
    raise ValueError(f"fmt must be 'probe' or 'dim', got {fmt!r}")


def _find_module(model, suffix: str, layer: int):
    for name, module in model.named_modules():
        if name.endswith(suffix) and _extract_layer_idx(name) == layer:
            return name, module
    raise ValueError(f"No module matching '*{suffix}' at layer {layer}")


class ActivationSteerer:
    """Attach one steering hook, generate with it active, then clean up."""

    def __init__(self, model, tokenizer, model_config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.handles = []
        # Model-agnostic end-of-turn terminators -- shared with the sampled
        # utils.inference.generate_batch path so both generation routes stop
        # identically for every registered model family.
        from utils.inference import resolve_terminators
        self.terminators = resolve_terminators(model, tokenizer)

    def attach(self, component: str, layer: int, vector: torch.Tensor, alpha: float, head: int = None):
        device = next(self.model.parameters()).device
        vector = vector.to(device)

        if component == "mha":
            if head is None:
                raise ValueError("component='mha' requires a head index")
            n_heads = self.model_config["n_heads"]
            head_dim = self.model_config["head_dim"]
            full_vec = torch.zeros(n_heads * head_dim, device=device)
            full_vec[head * head_dim : (head + 1) * head_dim] = alpha * vector

            _, module = _find_module(self.model, self.model_config["mha_hook"], layer)

            def pre_hook(m, inp, v=full_vec):
                x = inp[0]
                return (x + v.to(x.dtype),) + inp[1:]

            self.handles.append(module.register_forward_pre_hook(pre_hook))

        elif component == "mlp":
            _, module = _find_module(self.model, self.model_config["mlp_hook"], layer)

            def hook(m, inp, out, v=alpha * vector):
                return out + v.to(out.dtype)

            self.handles.append(module.register_forward_hook(hook))

        elif component == "residual":
            layer_name, _ = _find_module(self.model, self.model_config["mha_hook"], layer)
            layer_module_name = layer_name[: -len("." + self.model_config["mha_hook"])]
            layer_module = self.model.get_submodule(layer_module_name)

            def hook(m, inp, out, v=alpha * vector):
                if isinstance(out, tuple):
                    return (out[0] + v.to(out[0].dtype),) + out[1:]
                return out + v.to(out.dtype)

            self.handles.append(layer_module.register_forward_hook(hook))

        else:
            raise ValueError(f"component must be 'mha', 'mlp', or 'residual', got {component!r}")

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Greedy generation from a FULLY RENDERED chat prompt (including BOS,
        e.g. from build_chat_prompt) -- tokenized with add_special_tokens=False
        so the template's own <|begin_of_text|> isn't doubled."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        if len(new_tokens) and int(new_tokens[-1]) not in self.terminators:
            print(f"  WARNING: generation hit the max_new_tokens={max_new_tokens} cap "
                  "without emitting an end-of-turn token -- response is INCOMPLETE "
                  "(thinking models need a much larger budget).")
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(self, prompts: list, max_new_tokens: int = 150, batch_size: int = 8) -> list:
        """
        Same greedy, fully-rendered-chat-prompt contract as generate() (prompts
        must already include BOS; tokenized with add_special_tokens=False), but
        left-padded and chunked by batch_size for throughput. Requires
        tokenizer.padding_side == "left" (set by utils.model.load_model_and_tokenizer)
        -- right-padding would corrupt position ids for every prompt but the
        longest in a chunk under a causal LM.
        """
        if self.tokenizer.padding_side != "left":
            raise ValueError(
                "generate_batch requires tokenizer.padding_side == 'left' for correct "
                "batched causal-LM generation; got 'right'."
            )
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        responses = []
        n_truncated = 0
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            inputs = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024,
                add_special_tokens=False,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=self.terminators,
                    pad_token_id=pad_id,
                )
            input_len = inputs["input_ids"].shape[1]
            for i in range(output_ids.shape[0]):
                new_tokens = output_ids[i, input_len:]
                # A finished row ends with an end-of-turn token, or right-pad
                # after it; a row still mid-generation at the cap ends with an
                # ordinary token -- that response is incomplete.
                if len(new_tokens) and int(new_tokens[-1]) not in self.terminators and int(new_tokens[-1]) != pad_id:
                    n_truncated += 1
                responses.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        if n_truncated:
            print(f"  WARNING: {n_truncated}/{len(prompts)} generations hit the max_new_tokens={max_new_tokens} "
                  "cap without an end-of-turn token -- those responses are INCOMPLETE "
                  "(thinking models need a much larger budget).")
        return responses

    def cleanup(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
