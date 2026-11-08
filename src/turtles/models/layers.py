"""Transformer building blocks shared by UPT and CALM-PDE.

All blocks are pre-norm and optionally conditioned on a vector (PDE parameters)
through DiT-style adaLN-zero modulation, which starts every block at identity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint


class SincosEmbedding(nn.Module):
    """NeRF-style sine/cosine embedding of continuous coordinates.

    ``dim`` must be divisible by ``2 * ndim``; frequencies follow a geometric
    schedule from 1 to ``max_freq``.
    """

    def __init__(self, dim: int, ndim: int, max_freq: float = 100.0):
        super().__init__()
        if dim % (2 * ndim) != 0:
            raise ValueError(f"dim ({dim}) must be divisible by 2*ndim ({2 * ndim})")
        half = dim // (2 * ndim)
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half))
        self.register_buffer("freqs", freqs, persistent=False)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., ndim) -> (..., dim)``."""
        args = x.unsqueeze(-1) * self.freqs
        return torch.cat([args.sin(), args.cos()], dim=-1).flatten(-2)


class AdaLNModulation(nn.Module):
    """adaLN-zero: a condition vector -> six ``(B, 1, dim)`` modulation tensors."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return ``(shift_a, scale_a, gate_a, shift_m, scale_m, gate_m)``."""
        return self.proj(cond).unsqueeze(1).chunk(6, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x * (1 + scale) + shift``."""
    return x * (1 + scale) + shift


def feed_forward(dim: int, mlp_ratio: float) -> nn.Sequential:
    """Two-layer GELU MLP."""
    hidden = int(dim * mlp_ratio)
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))


class SelfAttention(nn.Module):
    """Multi-head self-attention on top of ``F.scaled_dot_product_attention``."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, dim) -> (B, N, dim)``."""
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to a key/value sequence."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """``q (B, Lq, dim), kv (B, Lk, dim) -> (B, Lq, dim)``."""
        B, Lq, D = q.shape
        Lk = kv.shape[1]
        q_h = self.q_proj(q).reshape(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        kv_h = self.kv_proj(kv).reshape(B, Lk, 2, self.num_heads, self.head_dim)
        k, v = kv_h.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(
            q_h, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(B, Lq, D))


class TransformerBlock(nn.Module):
    """Pre-norm self-attention + MLP with optional adaLN-zero conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        cond_dim: int | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        affine = cond_dim is None
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=affine)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=affine)
        self.mlp = feed_forward(dim, mlp_ratio)
        self.modulation = AdaLNModulation(dim, cond_dim) if cond_dim is not None else None
        self.use_checkpoint = use_checkpoint

    def _forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if self.modulation is not None and cond is not None:
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(cond)
            x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a))
            return x + gate_m * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, N, dim) [+ cond (B, cond_dim)] -> (B, N, dim)``."""
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, cond, use_reentrant=False)
        return self._forward(x, cond)


class _CrossBlock(nn.Module):
    """Cross-attention + MLP on a query sequence, with optional adaLN-zero conditioning."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, cond_dim: int | None):
        super().__init__()
        affine = cond_dim is None
        self.norm_q = nn.LayerNorm(dim, elementwise_affine=affine)
        self.norm_kv = nn.LayerNorm(dim, elementwise_affine=affine)
        self.attn = CrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=affine)
        self.mlp = feed_forward(dim, mlp_ratio)
        self.modulation = AdaLNModulation(dim, cond_dim) if cond_dim is not None else None

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``q (B, Lq, dim), kv (B, Lk, dim) -> (B, Lq, dim)``."""
        if self.modulation is not None and cond is not None:
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(cond)
            q = q + gate_a * self.attn(modulate(self.norm_q(q), shift_a, scale_a), self.norm_kv(kv))
            return q + gate_m * self.mlp(modulate(self.norm2(q), shift_m, scale_m))
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv))
        return q + self.mlp(self.norm2(q))


class PerceiverPooling(_CrossBlock):
    """Learned query tokens cross-attend to the inputs (perceiver pooling)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_queries: int,
        mlp_ratio: float,
        cond_dim: int | None,
    ):
        super().__init__(dim, num_heads, mlp_ratio, cond_dim)
        self.query = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

    def forward(self, kv: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        """``kv (B, Lk, dim) -> (B, num_queries, dim)``."""
        return super().forward(self.query.expand(kv.shape[0], -1, -1), kv, cond)


class PerceiverDecoder(_CrossBlock):
    """External query tokens (embedded coordinates) cross-attend to the latent tokens."""


class ParamConditioner(nn.Module):
    """PDE parameters ``(B, P)`` -> condition vector ``(B, cond_dim)``."""

    def __init__(self, num_params: int, dim: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_params, dim * 4), nn.GELU(), nn.Linear(dim * 4, cond_dim), nn.GELU()
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """``(B, P) -> (B, cond_dim)``."""
        return self.net(params)


def init_transformer_weights(module: nn.Module) -> None:
    """Truncated-normal linear weights, zero biases, unit LayerNorm (apply with ``model.apply``)."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if module.weight is not None:
            nn.init.ones_(module.weight)


def normalize_to_unit_box(coords: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample min-max normalization of ``(B, N, D)`` coordinates to ``[0, 1]^D``."""
    with torch.autocast(device_type=coords.device.type, enabled=False):
        c = coords.float()
        lo = c.amin(dim=1, keepdim=True)
        span = (c.amax(dim=1, keepdim=True) - lo).clamp_min(eps)
        out = (c - lo) / span
    return out.to(coords.dtype)
