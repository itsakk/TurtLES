"""Transolver++ (Luo et al., 2025) for unstructured-mesh forecasting.

A monolithic transformer: every node is a token; each block aggregates the ``N``
tokens into ``slice_num`` physics-aware slices per head (Gumbel-softmax routing),
runs self-attention on the slices and scatters the result back to the nodes.
Input and output cardinality are equal by construction, so full-mesh inference
uses the tile partition of :mod:`turtles.evaluation.predict` (no ``infer_full``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint

from turtles.models.base import config_from_mapping

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


@dataclass
class Config:
    """Transolver++ hyperparameters."""

    space_dim: int = 3
    fun_dim: int = 5
    out_dim: int = 5
    num_steps: int = 1
    n_layers: int = 8
    n_hidden: int = 512
    n_head: int = 8
    mlp_ratio: int = 4
    slice_num: int = 128
    dropout: float = 0.0
    act: str = "gelu"
    num_params: int = 0
    use_checkpoint: bool = True


class MLP(nn.Module):
    """Linear -> act -> [residual hidden layers] -> Linear."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 0,
        act: str = "gelu",
        residual: bool = True,
    ):
        super().__init__()
        act_cls = _ACTIVATIONS[act]
        self.residual = residual
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), act_cls())
        self.hidden = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act_cls()) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP on the last dimension."""
        x = self.input_proj(x)
        for layer in self.hidden:
            x = layer(x) + x if self.residual else layer(x)
        return self.output_proj(x)


def _gumbel_softmax(logits: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    # Noise is sampled in fp32 so the clamp bounds survive bf16 autocast.
    u = torch.rand_like(logits, dtype=torch.float32).clamp(1e-8, 1 - 1e-8)
    gumbel = -torch.log(-torch.log(u)).to(logits.dtype)
    return F.softmax((logits + gumbel) / tau, dim=-1)


class PhysicsSliceAttention(nn.Module):
    """Aggregate ``N`` tokens into ``G`` learned slices, attend over slices, scatter back."""

    def __init__(self, dim: int, heads: int, dropout: float, slice_num: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"n_hidden ({dim}) must be divisible by n_head ({heads})")
        self.heads = heads
        self.dim_head = dim // heads
        self.input_proj = nn.Linear(dim, dim)
        self.slice_proj = nn.Linear(self.dim_head, slice_num)
        nn.init.orthogonal_(self.slice_proj.weight)
        self.temperature_proj = nn.Sequential(
            nn.Linear(self.dim_head, slice_num), nn.GELU(), nn.Linear(slice_num, 1), nn.GELU()
        )
        self.bias = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, dim) -> (B, N, dim)``."""
        B, N, _ = x.shape
        x_h = self.input_proj(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
        tau = torch.clamp(self.temperature_proj(x_h) + self.bias, min=0.01)
        weights = _gumbel_softmax(self.slice_proj(x_h), tau)  # (B, H, N, G)
        slice_norm = weights.sum(dim=2)  # (B, H, G)
        slice_tokens = torch.einsum("bhnc,bhng->bhgc", x_h, weights)
        slice_tokens = slice_tokens / slice_norm.clamp_min(1e-5).unsqueeze(-1)
        attn = F.scaled_dot_product_attention(
            self.q(slice_tokens), self.k(slice_tokens), self.v(slice_tokens)
        )
        out = torch.einsum("bhgc,bhng->bhnc", attn, weights)  # (B, H, N, dh)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head)
        return self.out_proj(out)


class TransolverBlock(nn.Module):
    """Slice attention + feed-forward; the last block also projects to ``out_dim``."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        act: str,
        mlp_ratio: int,
        slice_num: int,
        out_dim: int,
        last: bool = False,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.last = last
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsSliceAttention(dim, heads, dropout, slice_num)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, dim * mlp_ratio, dim, n_layers=0, act=act, residual=False)
        if last:
            self.norm3 = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, out_dim)

    def _maybe_checkpoint(self, fn, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, dim) -> (B, N, dim)`` (or ``(B, N, out_dim)`` for the last block)."""
        x = self._maybe_checkpoint(self.attn, self.norm1(x)) + x
        x = self._maybe_checkpoint(self.ffn, self.norm2(x)) + x
        if self.last:
            return self.head(self.norm3(x))
        return x


class Model(nn.Module):
    """Transolver++ evolution operator."""

    def __init__(self, cfg: Mapping[str, Any] | Config):
        super().__init__()
        c = config_from_mapping(Config, cfg)
        self.c = c
        input_dim = (c.fun_dim + c.space_dim) * c.num_steps
        self.encoder = MLP(input_dim, c.n_hidden * 2, c.n_hidden, 0, c.act, residual=False)
        self.placeholder = nn.Parameter(torch.randn(c.n_hidden) / c.n_hidden)
        # PDE parameters enter as an additive bias on the encoded tokens.
        self.params_proj = nn.Linear(c.num_params, c.n_hidden) if c.num_params > 0 else None
        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    dim=c.n_hidden,
                    heads=c.n_head,
                    dropout=c.dropout,
                    act=c.act,
                    mlp_ratio=c.mlp_ratio,
                    slice_num=c.slice_num,
                    out_dim=c.out_dim,
                    last=(i == c.n_layers - 1),
                    use_checkpoint=c.use_checkpoint,
                )
                for i in range(c.n_layers)
            ]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        context: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor | None = None,
        query_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict at every input node (``query_coords`` is not supported: monolithic model)."""
        if query_coords is not None and query_coords.shape[1] != coords.shape[1]:
            raise ValueError(
                "Transolver++ predicts at its input nodes only (query_coords unsupported)"
            )
        B, T, N, _ = context.shape
        coords_t = coords.unsqueeze(1).expand(B, T, N, -1)
        x = torch.cat([context, coords_t], dim=-1).permute(0, 2, 1, 3).reshape(B, N, -1)
        h = self.encoder(x) + self.placeholder
        if params is not None and self.params_proj is not None:
            h = h + self.params_proj(params).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return h
