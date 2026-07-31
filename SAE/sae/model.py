"""
TopK sparse autoencoder (Gao et al., "Scaling and evaluating sparse
autoencoders"): a hard top-k mask over encoder pre-activations replaces the
usual L1 sparsity penalty. Dead features are kept alive via an auxiliary loss
that reconstructs the residual error from otherwise-dead features, rather
than by resampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, hidden_size: int, dict_size: int, k: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.k = k

        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.encoder = nn.Linear(hidden_size, dict_size)
        self.decoder = nn.Linear(dict_size, hidden_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))
            self.encoder.weight.copy_(self.decoder.weight.t())
        nn.init.zeros_(self.encoder.bias)

    def topk_mask(self, pre_acts: torch.Tensor) -> torch.Tensor:
        k = min(self.k, pre_acts.shape[-1])
        values, indices = torch.topk(pre_acts, k, dim=-1)
        mask = torch.zeros_like(pre_acts)
        mask.scatter_(-1, indices, values)
        return mask

    def encode(self, x: torch.Tensor):
        pre_acts = F.relu(self.encoder(x - self.pre_bias))
        code = self.topk_mask(pre_acts)
        return code, pre_acts

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.decoder(code) + self.pre_bias

    def forward(self, x: torch.Tensor):
        code, pre_acts = self.encode(x)
        recon = self.decode(code)
        return recon, code, pre_acts

    @torch.no_grad()
    def renormalize_decoder_(self):
        norms = self.decoder.weight.norm(dim=0, keepdim=True)
        self.decoder.weight.div_(norms.clamp_min(1e-8))

    def aux_loss(self, x: torch.Tensor, recon: torch.Tensor, pre_acts: torch.Tensor,
                 dead_mask: torch.Tensor, k_aux: int) -> torch.Tensor:
        """
        Reconstruct (x - recon) from the top-k_aux currently-dead features
        only, so features that haven't won top-k recently still get a
        gradient signal instead of dying permanently.
        """
        n_dead = int(dead_mask.sum().item())
        k_aux = min(k_aux, n_dead)
        if k_aux <= 0:
            return recon.new_zeros(())

        dead_acts = pre_acts.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
        values, indices = torch.topk(dead_acts, k_aux, dim=-1)
        aux_code = torch.zeros_like(pre_acts)
        aux_code.scatter_(-1, indices, values.clamp_min(0))

        residual = (x - recon).detach()
        aux_recon = self.decoder(aux_code)
        return F.mse_loss(aux_recon, residual)
