"""CALM-PDE (Hagnberger et al., 2025) with the ECHO scaling improvements.

Three stages, end-to-end differentiable:

1. A continuous-convolution **encoder** lifts the variable mesh ``(B, N, T*C)`` onto a
   fixed latent point cloud ``(B, L, dim)``. With global coordinate bounds the cloud
   is a regular ``n^3`` grid shared by all samples; otherwise coordinates are
   normalized per sample and the cloud is a (learnable) random set.
2. A UPT-style **transformer** on the ``L`` latent tokens is the time-stepping
   operator, conditioned on PDE parameters through adaLN-zero.
3. A continuous-convolution **decoder** maps the latent tokens back to arbitrary
   query points (the mesh nodes). Its fused CUDA path is memory-flat in the
   number of queries (see :mod:`turtles.models.calm_pde_kernels`).

The encoder/decoder split gives :meth:`Model.infer_full` (encode the full mesh once,
decode in chunks) and query subsampling during training (``query_coords``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from turtles.models.base import config_from_mapping, flatten_context
from turtles.models.calm_pde_kernels import _decode_block, fused_decode_autograd, mlp_params
from turtles.models.knn import knn
from turtles.models.layers import (
    ParamConditioner,
    SincosEmbedding,
    TransformerBlock,
    init_transformer_weights,
    normalize_to_unit_box,
)


@dataclass
class Config:
    """CALM-PDE hyperparameters."""

    space_dim: int = 3
    fun_dim: int = 5
    out_dim: int = 5
    num_steps: int = 1

    num_latent_points: int = 4096
    knn_k: int = 32
    kernel_hidden: int = 64  # power of two >= 16 to use the fused CUDA decoder
    temperature: float = 1.0
    learn_latent_grid: bool = False
    # Global coordinate bounds -> fixed regular latent grid + global normalization.
    coord_min: list[float] | None = None
    coord_max: list[float] | None = None

    dim: int = 384
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    depth: int = 8
    pos_embed_max_freq: float = 100.0

    num_params: int = 0
    cond_dim: int | None = None
    use_checkpoint: bool = True

    decoder_bwd_chunk: int = 8192
    decoder_query_chunk_size: int | None = 32768


class ContinuousConv(nn.Module):
    """Continuous k-NN convolution with an MLP kernel on ``(delta, |delta|^2)``.

    For every query the ``K`` nearest sources are gathered, weighted by
    ``softmax(-|delta|^2 / T)`` and contracted with the kernel MLP output
    (``in_channels x out_channels`` per neighbour), using the algebraic factorisation
    of :func:`turtles.models.calm_pde_kernels._decode_block`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        num_neighbors: int = 16,
        hidden_channels: int = 64,
        temperature: float = 1.0,
        query_chunk_size: int | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.num_neighbors = num_neighbors
        self.temperature = temperature
        self.eps = 1e-8
        self.query_chunk_size = query_chunk_size
        self.use_checkpoint = use_checkpoint
        self.kernel_mlp = nn.Sequential(
            nn.Linear(spatial_dim + 1, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, in_channels * out_channels),
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    @staticmethod
    def _expand(pos: torch.Tensor, B: int) -> torch.Tensor:
        return pos.unsqueeze(0).expand(B, -1, -1) if pos.dim() == 2 else pos

    def _forward_chunk(
        self, x: torch.Tensor, src_pos: torch.Tensor, query_pos: torch.Tensor
    ) -> torch.Tensor:
        K = min(self.num_neighbors, x.shape[1])
        # Neighbour ranking in fp32: bf16 distances tie at large N and reorder neighbours.
        idx = knn(query_pos, src_pos, K)
        w1, b1, w2, b2, w3, b3 = mlp_params(self.kernel_mlp)
        return _decode_block(
            query_pos, src_pos, x, idx, w1, b1, w2, b2, w3, b3,
            in_channels=self.in_channels, out_channels=self.out_channels,
            temperature=self.temperature, eps=self.eps,
        )  # fmt: skip

    def forward(
        self, x: torch.Tensor, src_pos: torch.Tensor, query_pos: torch.Tensor
    ) -> torch.Tensor:
        """``x (B, M, Cin)`` at ``src_pos`` -> ``(B, Q, Cout)`` at ``query_pos``."""
        B = x.shape[0]
        src_pos = self._expand(src_pos, B)
        query_pos = self._expand(query_pos, B)
        Q = query_pos.size(1)
        chunk = min(self.query_chunk_size or Q, Q)
        if chunk >= Q:
            return self._forward_chunk(x, src_pos, query_pos) + self.bias
        use_ckpt = self.use_checkpoint and torch.is_grad_enabled()
        outs = []
        for s in range(0, Q, chunk):
            q_chunk = query_pos[:, s : s + chunk].contiguous()
            if use_ckpt:
                outs.append(
                    checkpoint(self._forward_chunk, x, src_pos, q_chunk, use_reentrant=False)
                )
            else:
                outs.append(self._forward_chunk(x, src_pos, q_chunk))
        return torch.cat(outs, dim=1) + self.bias


class ContinuousConvEncoder(ContinuousConv):
    """Variable source mesh -> fixed latent cloud (the query set is a parameter)."""

    def __init__(
        self, *args, num_query_points: int, learn_grid: bool, init_query_pos=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if init_query_pos is None:
            init_query_pos = torch.rand(num_query_points, self.spatial_dim)
        self.latent_grid = nn.Parameter(init_query_pos, requires_grad=learn_grid)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """``x (B, N, Cin)`` at ``pos (B, N, D)`` -> ``(B, L, Cout)`` on the latent grid."""
        return super().forward(x, src_pos=pos, query_pos=self.latent_grid)


class ContinuousConvDecoder(ContinuousConv):
    """Latent cloud (shared source) -> arbitrary query points, fused on CUDA."""

    def __init__(self, *args, use_fused: bool = True, bwd_chunk: int = 8192, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fused = use_fused
        self.bwd_chunk = bwd_chunk
        self.latent_grid: nn.Parameter | None = None  # tied to the encoder's grid by Model

    def forward(self, x: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """``x (B, L, Cin)`` on the latent grid -> ``(B, Q, Cout)`` at ``query_pos``."""
        assert self.latent_grid is not None, "decoder latent grid is not set"
        if self.use_fused and x.is_cuda:
            out = fused_decode_autograd(
                self._expand(query_pos, x.shape[0]), self.latent_grid, x, self.kernel_mlp,
                in_channels=self.in_channels, out_channels=self.out_channels,
                num_neighbors=self.num_neighbors, temperature=self.temperature,
                bwd_chunk=self.bwd_chunk,
            )  # fmt: skip
            return out + self.bias
        return super().forward(x, src_pos=self.latent_grid, query_pos=query_pos)


def regular_unit_grid(n_per_axis: int, ndim: int) -> torch.Tensor:
    """Regular ``n^ndim`` lattice in ``[0, 1]^ndim`` as ``(n^ndim, ndim)``."""
    axis = torch.linspace(0.0, 1.0, n_per_axis)
    return torch.cartesian_prod(*([axis] * ndim)).reshape(-1, ndim)


class Model(nn.Module):
    """CALM-PDE evolution operator: encoder -> latent transformer -> decoder."""

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

        self.use_global_bounds = c.coord_min is not None and c.coord_max is not None
        if self.use_global_bounds:
            cmin = torch.tensor(c.coord_min, dtype=torch.float32)
            cmax = torch.tensor(c.coord_max, dtype=torch.float32)
            if cmin.numel() != c.space_dim or cmax.numel() != c.space_dim:
                raise ValueError(f"coord_min/coord_max must have length space_dim={c.space_dim}")
            self.register_buffer("coord_min", cmin, persistent=True)
            self.register_buffer("coord_max", cmax, persistent=True)
            n = round(c.num_latent_points ** (1.0 / c.space_dim))
            if n**c.space_dim != c.num_latent_points:
                raise ValueError(
                    f"With coordinate bounds, num_latent_points={c.num_latent_points} must be "
                    f"n**{c.space_dim} for an integer n, e.g. {n**c.space_dim} or {(n + 1) ** c.space_dim}"
                )
            init_grid, learn_grid = regular_unit_grid(n, c.space_dim), False
        else:
            init_grid, learn_grid = None, c.learn_latent_grid

        self.encoder = ContinuousConvEncoder(
            c.fun_dim * c.num_steps, c.dim, c.space_dim,
            num_neighbors=c.knn_k, hidden_channels=c.kernel_hidden, temperature=c.temperature,
            num_query_points=c.num_latent_points, learn_grid=learn_grid, init_query_pos=init_grid,
        )  # fmt: skip
        self.latent_pos_embed = SincosEmbedding(c.dim, c.space_dim, max_freq=c.pos_embed_max_freq)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    c.dim, c.num_heads, c.mlp_ratio, c.dropout, block_cond, c.use_checkpoint
                )
                for _ in range(c.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(c.dim)
        self.decoder = ContinuousConvDecoder(
            c.dim, c.out_dim, c.space_dim,
            num_neighbors=c.knn_k, hidden_channels=c.kernel_hidden, temperature=c.temperature,
            query_chunk_size=c.decoder_query_chunk_size, use_checkpoint=c.use_checkpoint,
            bwd_chunk=c.decoder_bwd_chunk,
        )  # fmt: skip
        self.apply(init_transformer_weights)
        # Encoder queries and decoder sources are the same latent cloud.
        self.decoder.latent_grid = self.encoder.latent_grid

    # ------------------------------------------------------------------ pieces

    def _to_unit(self, coords: torch.Tensor) -> torch.Tensor:
        if not self.use_global_bounds:
            return normalize_to_unit_box(coords)
        with torch.autocast(device_type=coords.device.type, enabled=False):
            span = (self.coord_max - self.coord_min).clamp_min(1e-6)
            out = (coords.float() - self.coord_min) / span
        return out.to(coords.dtype)

    def _condition(self, params: torch.Tensor | None) -> torch.Tensor | None:
        if self.conditioner is None or params is None:
            return None
        return self.conditioner(params)

    def encode(
        self, context: torch.Tensor, coords_unit: torch.Tensor, cond: torch.Tensor | None
    ) -> torch.Tensor:
        """Mesh ``(B, T, N, C)`` with unit-box coords -> latent tokens ``(B, L, dim)``."""
        h = self.encoder(flatten_context(context), coords_unit)
        h = h + self.latent_pos_embed(self.encoder.latent_grid)
        for block in self.blocks:
            h = block(h, cond)
        return self.final_norm(h)

    # ------------------------------------------------------------------ contract

    def forward(
        self,
        context: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor | None = None,
        query_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict at ``query_coords`` (default: the input nodes); the encoder sees all nodes."""
        coords_unit = self._to_unit(coords)
        latent = self.encode(context, coords_unit, self._condition(params))
        queries = coords_unit if query_coords is None else self._to_unit(query_coords)
        return self.decoder(latent, queries)

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
        """Full-mesh inference: encode the whole mesh once, decode ``query_chunk_size`` nodes at a time."""
        B, _, N, _ = context.shape
        if input_subset_size is not None and input_subset_size < N:
            perm = torch.randperm(N, device=context.device)[:input_subset_size]
            context_in, coords_in = context[:, :, perm].contiguous(), coords[:, perm].contiguous()
        else:
            context_in, coords_in = context, coords
        latent = self.encode(context_in, self._to_unit(coords_in), self._condition(params))
        coords_unit = self._to_unit(coords)
        if query_chunk_size >= N:
            return self.decoder(latent, coords_unit)
        out = torch.empty(B, N, self.c.out_dim, device=context.device, dtype=context.dtype)
        for start in range(0, N, query_chunk_size):
            end = min(start + query_chunk_size, N)
            out[:, start:end] = self.decoder(latent, coords_unit[:, start:end].contiguous())
        return out
