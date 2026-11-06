"""Universal Physics Transformer (Alkin et al., 2024) for unstructured-mesh forecasting.

Pipeline:

1. Supernode pooling: ``K`` mesh nodes are drawn as supernodes; each gathers its
   ``k`` nearest mesh nodes (runtime k-NN) through a message-passing MLP.
2. Encoder transformer on the ``K`` supernode tokens.
3. Perceiver pooling to ``L`` learned latent tokens, then the latent transformer
   (the time-stepping operator) and the decoder transformer.
4. Perceiver decoding: sincos-embedded query coordinates cross-attend to the
   latent tokens; a linear head emits the per-node output.

The encoder/decoder split allows :meth:`Model.infer_full`: the full mesh is
encoded once and queries are decoded in chunks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from turtles.models.base import config_from_mapping, flatten_context
from turtles.models.knn import knn
from turtles.models.layers import (
    ParamConditioner,
    PerceiverDecoder,
    PerceiverPooling,
    SincosEmbedding,
    TransformerBlock,
    init_transformer_weights,
)


@dataclass
class Config:
    """UPT hyperparameters."""

    space_dim: int = 3
    fun_dim: int = 5
    out_dim: int = 5
    num_steps: int = 1

    dim: int = 384
    num_heads: int = 6
    perc_num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    num_supernodes: int = 16384
    knn_k: int = 32
    num_latent_tokens: int = 4096
    pos_embed_max_freq: float = 100.0

    enc_depth: int = 2
    latent_depth: int = 4
    dec_depth: int = 2

    num_params: int = 0
    cond_dim: int | None = None
    use_checkpoint: bool = True


class SupernodePooling(nn.Module):
    """Pool mesh features onto ``K`` supernodes via k-NN message passing.

    Features and positions are gathered at the ``K x k`` neighbour nodes *before*
    the projections, so the cost is independent of the mesh size ``N`` (only the
    k-NN search touches all nodes).
    """

    def __init__(self, input_dim: int, hidden_dim: int, ndim: int, knn_k: int, max_freq: float):
        super().__init__()
        self.knn_k = knn_k
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_embed = SincosEmbedding(hidden_dim, ndim, max_freq=max_freq)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, mesh_pos: torch.Tensor, super_pos: torch.Tensor
    ) -> torch.Tensor:
        """``x (B, N, F), mesh_pos (B, N, D), super_pos (B, K, D) -> (B, K, hidden)``."""
        B = super_pos.shape[0]
        idx = knn(super_pos, mesh_pos, self.knn_k)  # (B, K, k)
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1)
        x_nb = x[batch_idx, idx]  # (B, K, k, F)
        pos_nb = mesh_pos[batch_idx, idx]  # (B, K, k, D)
        h_nb = self.input_proj(x_nb) + self.pos_embed(pos_nb)
        h_super = self.pos_embed(super_pos).unsqueeze(2).expand(-1, -1, idx.shape[2], -1)
        return self.message(torch.cat([h_nb, h_super], dim=-1)).mean(dim=2)


class Model(nn.Module):
    """UPT evolution operator."""

    supports_query_coords = True

    def __init__(self, cfg: Mapping[str, Any] | Config):
        super().__init__()
        c = config_from_mapping(Config, cfg)
        self.c = c
        cond_dim = c.cond_dim if c.cond_dim is not None else 4 * c.dim
        self.use_cond = c.num_params > 0
        block_cond = cond_dim if self.use_cond else None
        self.conditioner = (
            ParamConditioner(c.num_params, c.dim, cond_dim) if self.use_cond else None
        )

        self.pool = SupernodePooling(
            c.fun_dim * c.num_steps, c.dim, c.space_dim, c.knn_k, c.pos_embed_max_freq
        )

        def stack(depth: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    TransformerBlock(
                        c.dim, c.num_heads, c.mlp_ratio, c.dropout, block_cond, c.use_checkpoint
                    )
                    for _ in range(depth)
                ]
            )

        self.enc_blocks = stack(c.enc_depth)
        self.perceiver_pool = PerceiverPooling(
            c.dim, c.perc_num_heads, c.num_latent_tokens, c.mlp_ratio, block_cond
        )
        self.latent_blocks = stack(c.latent_depth)
        self.dec_blocks = stack(c.dec_depth)

        self.query_embed = SincosEmbedding(c.dim, c.space_dim, max_freq=c.pos_embed_max_freq)
        self.query_mlp = nn.Sequential(
            nn.Linear(c.dim, 4 * c.dim),
            nn.GELU(),
            nn.Linear(4 * c.dim, 4 * c.dim),
            nn.GELU(),
            nn.Linear(4 * c.dim, c.dim),
        )
        self.perceiver_decode = PerceiverDecoder(c.dim, c.perc_num_heads, c.mlp_ratio, block_cond)
        self.head_norm = nn.LayerNorm(c.dim)
        self.head = nn.Linear(c.dim, c.out_dim)
        self.apply(init_transformer_weights)

    # ------------------------------------------------------------------ pieces

    def _condition(self, params: torch.Tensor | None) -> torch.Tensor | None:
        if self.conditioner is None or params is None:
            return None
        return self.conditioner(params)

    def _sample_supernodes(self, coords: torch.Tensor) -> torch.Tensor:
        B, N, _ = coords.shape
        K = min(self.c.num_supernodes, N)
        perm = torch.argsort(torch.rand(B, N, device=coords.device), dim=1)[:, :K]
        return torch.gather(coords, 1, perm.unsqueeze(-1).expand(-1, -1, coords.shape[-1]))

    def encode(
        self, context: torch.Tensor, coords: torch.Tensor, cond: torch.Tensor | None
    ) -> torch.Tensor:
        """Mesh ``(B, T, N, C)`` -> latent tokens ``(B, L, dim)`` after the decoder blocks."""
        x = flatten_context(context)
        h = self.pool(x, coords, self._sample_supernodes(coords))
        for block in self.enc_blocks:
            h = block(h, cond)
        h = self.perceiver_pool(h, cond)
        for block in self.latent_blocks:
            h = block(h, cond)
        for block in self.dec_blocks:
            h = block(h, cond)
        return h

    def decode(
        self, latent: torch.Tensor, query_coords: torch.Tensor, cond: torch.Tensor | None
    ) -> torch.Tensor:
        """Latent tokens ``(B, L, dim)`` -> fields at ``query_coords`` ``(B, Q, out_dim)``."""
        q = self.query_mlp(self.query_embed(query_coords))
        out = self.perceiver_decode(q, latent, cond)
        return self.head(self.head_norm(out))

    # ------------------------------------------------------------------ contract

    def forward(
        self,
        context: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor | None = None,
        query_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the next frame at ``query_coords`` (default: the input nodes)."""
        cond = self._condition(params)
        latent = self.encode(context, coords, cond)
        return self.decode(latent, coords if query_coords is None else query_coords, cond)

    @torch.no_grad()
    def infer_full(
        self,
        context: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor | None = None,
        *,
        input_subset_size: int | None = None,
        query_chunk_size: int = 65_536,
    ) -> torch.Tensor:
        """Full-mesh inference: encode once, decode ``query_chunk_size`` nodes at a time."""
        B, _, N, _ = context.shape
        if input_subset_size is not None and input_subset_size < N:
            perm = torch.randperm(N, device=context.device)[:input_subset_size]
            context_in, coords_in = context[:, :, perm].contiguous(), coords[:, perm].contiguous()
        else:
            context_in, coords_in = context, coords
        cond = self._condition(params)
        latent = self.encode(context_in, coords_in, cond)
        out = torch.empty(B, N, self.c.out_dim, device=context.device, dtype=context.dtype)
        for start in range(0, N, query_chunk_size):
            end = min(start + query_chunk_size, N)
            out[:, start:end] = self.decode(latent, coords[:, start:end].contiguous(), cond)
        return out
