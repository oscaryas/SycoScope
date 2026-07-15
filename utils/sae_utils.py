"""
Top-K sparse autoencoder model + loading helpers for the cached-activations
artifact. Pure building blocks -- no training loop, no CLI. SAE/pipeline/
train_sae.py is the script that actually uses these to produce trained
weights; this module never touches the LLM or SAE/pipeline/cache_activations.py.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTIVATIONS_DIR = REPO_ROOT / "SAE" / "results" / "activations"
DEFAULT_TRAINED_SAE_DIR = REPO_ROOT / "SAE" / "results" / "trained_sae"


def topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """Zero all but the top-k (post-ReLU) entries per row; only non-negative
    pre-activations can fire, matching standard top-k SAE convention."""
    vals, idx = torch.topk(F.relu(z), k, dim=-1)
    return torch.zeros_like(z).scatter_(-1, idx, vals)


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, n_latents: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.n_latents = n_latents
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        W_dec = torch.randn(d_in, n_latents)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.normalize_decoder_()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pre = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        z = topk_mask(z_pre, self.k)
        return z, z_pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, _ = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalize_decoder_(self):
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))


def load_layer_array(layer: int, activations_dir: Path = DEFAULT_ACTIVATIONS_DIR) -> np.ndarray:
    return np.load(activations_dir / f"layer_{layer:02d}.npy")


def compute_global_center(layer: int, activations_dir: Path = DEFAULT_ACTIVATIONS_DIR) -> np.ndarray:
    means = np.load(activations_dir / f"response_means_layer_{layer:02d}.npy")
    return means.astype(np.float32).mean(axis=0)


def sae_dir_name(layer: int, n_latents: int, k: int, centered: bool, seed: int) -> str:
    tag = "centered" if centered else "uncentered"
    return f"L{layer:02d}_n{n_latents}_k{k}_{tag}_s{seed}"
