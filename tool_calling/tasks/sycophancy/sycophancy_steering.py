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
from utils.inference import generate_from_rendered


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
        self.last_truncated: list[bool] = []

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
        e.g. from build_chat_prompt), with whatever hooks attach() has
        registered still active."""
        return self.generate_batch([prompt], max_new_tokens=max_new_tokens, batch_size=1)[0]

    def generate_batch(self, prompts: list, max_new_tokens: int = 150, batch_size: int = 8) -> list:
        """
        Same greedy, fully-rendered-chat-prompt contract as generate() (prompts
        must already include BOS), with whatever hooks attach() has registered
        still active. Delegates all tokenize/generate/decode/terminator-
        resolution mechanics to utils.inference.generate_from_rendered, so
        steered and unsteered generation can never diverge again the way they
        did before this fix. Sets self.last_truncated: list[bool], one entry
        per prompt, flagging responses that hit max_new_tokens without an
        end-of-turn token.
        """
        responses, truncated = generate_from_rendered(
            self.model, self.tokenizer, prompts,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
        )
        self.last_truncated = truncated
        return responses

    def cleanup(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
