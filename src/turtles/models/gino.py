"""GINO (Li et al., 2023): adapter around the official ``neuralop.models.GINO``.

GNO encoder (mesh -> regular latent grid) -> FNO -> GNO decoder (grid -> queries).
Vendoring the blocks would drift from upstream, so this module only adapts the
TurtLES batching to the library:

* GINO assumes one geometry per call; the batch is looped over.
* Coordinates are min-max normalized to ``[0, 1]^3`` per sample so that the GNO
  radii are scale-free.
* PDE parameters condition the FNO through adaptive instance norm.
* The spectral convolutions do not support bf16 and break ``torch.compile``:
  autocast is disabled inside the call and Dynamo skips the whole adapter.
* neuraloperator's native ``segment_csr`` is a Python loop over every output node
  (about a minute per step at 524k nodes) and its native radius search materialises a
  dense query x data distance matrix. The adapter swaps in a vectorized reduction and
  a chunked radius search, so GINO is fast without Open3D ML ops or torch_scatter.

Install with ``pip install turtles[gino]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from turtles.models.base import config_from_mapping, flatten_context

try:
    from neuralop.models import GINO as _GINO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The GINO baseline needs the `neuraloperator` package: pip install 'turtles[gino]'"
    ) from exc


def fast_segment_csr(
    src: torch.Tensor, indptr: torch.Tensor, reduce: str, use_scatter: bool = True
):
    """Vectorized CSR segment reduction (drop-in for ``neuralop.layers.segment_csr``).

    ``src`` is ``(n_reps, C)`` or batched ``(B, n_reps, C)``; ``indptr`` holds the
    ``n_out + 1`` row splits. Returns ``(n_out, C)`` or ``(B, n_out, C)``.
    """
    del use_scatter
    if reduce not in ("mean", "sum"):
        raise ValueError("reduce must be 'mean' or 'sum'")
    splits = indptr[0] if indptr.ndim == 2 else indptr
    counts = (splits[1:] - splits[:-1]).long()
    n_out = counts.numel()
    batched = src.ndim == 3
    point_dim = 1 if batched else 0
    segment = torch.repeat_interleave(torch.arange(n_out, device=src.device), counts)
    out_shape = list(src.shape)
    out_shape[point_dim] = n_out
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    out.index_add_(point_dim, segment, src)
    if reduce == "mean":
        denom = counts.clamp_min(1).to(src.dtype)
        out = out / (denom.view(1, -1, 1) if batched else denom.view(-1, 1))
    return out


def chunked_radius_search(
    data: torch.Tensor, queries: torch.Tensor, radius: float, chunk: int = 8192
) -> dict[str, torch.Tensor]:
    """All ``data`` points within ``radius`` of each query, in CSR form, chunked over queries.

    Same output as ``neuralop.layers.neighbor_search.native_neighbor_search`` without the
    full ``(queries, data)`` distance matrix.
    """
    # Bound each distance block to ~1 GB in fp32 whatever the data size.
    chunk = max(1, min(int(chunk), 2**28 // max(data.shape[0], 1)))
    indices: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    for start in range(0, queries.shape[0], chunk):
        dists = torch.cdist(queries[start : start + chunk], data)
        mask = dists <= radius
        indices.append(mask.nonzero()[:, 1])
        counts.append(mask.sum(dim=1))
    count = torch.cat(counts)
    splits = torch.cat([count.new_zeros(1), torch.cumsum(count, dim=0)])
    return {"neighbors_index": torch.cat(indices).long(), "neighbors_row_splits": splits.long()}


def _patch_neuralop_kernels(gino: nn.Module) -> None:
    """Route the GNO blocks through the fast reduction and radius search above."""
    import neuralop.layers.integral_transform as integral_transform

    integral_transform.segment_csr = fast_segment_csr
    for block in (gino.gno_in, gino.gno_out):
        search = getattr(block, "neighbor_search", None)
        if search is not None and not getattr(search, "use_open3d", False):
            search.search_fn = chunked_radius_search


def _no_compile(fn):
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(recursive=True)(fn)
    return fn


@dataclass
class Config:
    """GINO hyperparameters (see ``neuralop.models.GINO`` for the semantics)."""

    space_dim: int = 3
    fun_dim: int = 5
    out_dim: int = 5
    num_steps: int = 1

    latent_grid_resolution: int = 12
    fno_n_modes: list[int] = field(default_factory=lambda: [6, 6, 6])
    fno_hidden_channels: int = 64
    fno_n_layers: int = 4
    fno_lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 4

    in_gno_radius: float = 0.033
    out_gno_radius: float = 0.033
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    in_gno_channel_mlp_hidden_layers: list[int] = field(default_factory=lambda: [80, 80, 80])
    out_gno_channel_mlp_hidden_layers: list[int] = field(default_factory=lambda: [128, 64])
    gno_embed_channels: int = 32

    num_params: int = 0
    fno_ada_in_features: int = 4

    gno_use_open3d: bool = False
    gno_use_torch_scatter: bool = False


class Model(nn.Module):
    """GINO evolution operator."""

    supports_query_coords = True

    def __init__(self, cfg: Mapping[str, Any] | Config):
        super().__init__()
        c = config_from_mapping(Config, cfg)
        if len(c.fno_n_modes) != c.space_dim:
            raise ValueError(
                f"fno_n_modes {c.fno_n_modes} must have length space_dim={c.space_dim}"
            )
        if any(m >= c.latent_grid_resolution // 2 + 1 for m in c.fno_n_modes):
            raise ValueError(
                f"fno_n_modes={c.fno_n_modes} exceed Nyquist for resolution {c.latent_grid_resolution}"
            )
        self.c = c
        self.use_cond = c.num_params > 0

        axis = torch.linspace(0.0, 1.0, c.latent_grid_resolution)
        grid = torch.stack(torch.meshgrid(*([axis] * c.space_dim), indexing="ij"), dim=-1)
        self.register_buffer("latent_grid", grid.unsqueeze(0), persistent=False)

        in_channels = c.fun_dim * c.num_steps
        kwargs: dict[str, Any] = dict(
            in_channels=in_channels,
            out_channels=c.out_dim,
            gno_coord_dim=c.space_dim,
            fno_n_modes=tuple(c.fno_n_modes),
            fno_hidden_channels=c.fno_hidden_channels,
            fno_n_layers=c.fno_n_layers,
            # In the linear GNO modes the kernel output multiplies f_y, so the FNO
            # input width is pinned to the field width; the lifting MLP expands it.
            fno_in_channels=in_channels,
            fno_lifting_channel_ratio=c.fno_lifting_channel_ratio,
            in_gno_radius=c.in_gno_radius,
            out_gno_radius=c.out_gno_radius,
            in_gno_transform_type=c.in_gno_transform_type,
            out_gno_transform_type=c.out_gno_transform_type,
            in_gno_channel_mlp_hidden_layers=list(c.in_gno_channel_mlp_hidden_layers),
            out_gno_channel_mlp_hidden_layers=list(c.out_gno_channel_mlp_hidden_layers),
            gno_embed_channels=c.gno_embed_channels,
            projection_channel_ratio=c.projection_channel_ratio,
            gno_use_open3d=c.gno_use_open3d,
            gno_use_torch_scatter=c.gno_use_torch_scatter,
        )
        if self.use_cond:
            kwargs.update(
                fno_norm="ada_in",
                fno_ada_in_features=c.fno_ada_in_features,
                fno_ada_in_dim=c.num_params,
            )
        self.gino = _GINO(**kwargs)
        _patch_neuralop_kernels(self.gino)
        # Encode-once / decode-in-chunks needs these upstream internals (neuralop 1.0.x).
        self.supports_infer_full = all(
            hasattr(self.gino, attr)
            for attr in (
                "gno_in",
                "gno_out",
                "latent_embedding",
                "projection",
                "in_coord_dim_reverse_order",
                "fno_hidden_channels",
                "out_gno_tanh",
            )
        )

    @staticmethod
    def _unit_box(coords: torch.Tensor) -> torch.Tensor:
        lo = coords.amin(dim=0, keepdim=True)
        hi = coords.amax(dim=0, keepdim=True)
        return (coords - lo) / (hi - lo).clamp_min(1e-6)

    @_no_compile
    def forward(
        self,
        context: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor | None = None,
        query_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the next frame at ``query_coords`` (default: the input nodes)."""
        B = context.shape[0]
        x = flatten_context(context).float()
        coords_f = coords.float()
        queries_f = coords_f if query_coords is None else query_coords.float()
        params_f = params.float() if params is not None else None
        outs = []
        with torch.autocast(device_type=context.device.type, enabled=False):
            for b in range(B):
                lo = coords_f[b].amin(dim=0, keepdim=True)
                span = (coords_f[b].amax(dim=0, keepdim=True) - lo).clamp_min(1e-6)
                geom = (coords_f[b] - lo) / span
                queries = (queries_f[b] - lo) / span
                kwargs: dict[str, Any] = dict(
                    input_geom=geom.unsqueeze(0),
                    latent_queries=self.latent_grid,
                    output_queries=queries,  # 2-D on purpose (some versions skip the squeeze)
                    x=x[b].unsqueeze(0),
                )
                if self.use_cond and params_f is not None:
                    kwargs["ada_in"] = params_f[b]
                outs.append(self.gino(**kwargs))
        return torch.cat(outs, dim=0).to(context.dtype)

    @torch.no_grad()
    @_no_compile
    def infer_full(
        self,
        context: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor | None = None,
        *,
        input_subset_size: int | None = None,
        query_chunk_size: int = 65_536,
    ) -> torch.Tensor:
        """Full-mesh inference for ``B = 1``: run encoder + FNO once, decode query chunks.

        Mirrors ``neuralop.models.GINO.forward`` step by step; unavailable (see
        ``supports_infer_full``) when the installed library lacks those internals.
        """
        if not self.supports_infer_full:
            raise NotImplementedError("installed neuraloperator does not expose GINO internals")
        if context.shape[0] != 1:
            raise ValueError("infer_full expects a batch of one trajectory")
        gino = self.gino
        N = coords.shape[1]
        with torch.autocast(device_type=context.device.type, enabled=False):
            geom_all = self._unit_box(coords[0].float())
            x = flatten_context(context).float()
            if input_subset_size is not None and input_subset_size < N:
                perm = torch.randperm(N, device=coords.device)[:input_subset_size]
                geom_in, x_in = geom_all[perm], x[:, perm]
            else:
                geom_in, x_in = geom_all, x
            latent_queries = self.latent_grid.squeeze(0)
            flat_queries = latent_queries.reshape(-1, latent_queries.shape[-1])

            in_p = gino.gno_in(y=geom_in, x=flat_queries, f_y=x_in)
            in_p = in_p.view((1, *latent_queries.shape[:-1], -1))
            ada_in = params[0].float() if (self.use_cond and params is not None) else None
            latent = gino.latent_embedding(in_p=in_p, ada_in=ada_in)
            latent = latent.permute(0, *gino.in_coord_dim_reverse_order, 1)
            latent = latent.reshape(1, -1, gino.fno_hidden_channels)
            if gino.out_gno_tanh in ("latent_embed", "both"):
                latent = torch.tanh(latent)

            out = torch.empty(1, N, self.c.out_dim, device=coords.device, dtype=torch.float32)
            for start in range(0, N, query_chunk_size):
                end = min(start + query_chunk_size, N)
                chunk = gino.gno_out(y=flat_queries, x=geom_all[start:end], f_y=latent)
                chunk = gino.projection(chunk.permute(0, 2, 1)).permute(0, 2, 1)
                out[:, start:end] = chunk
        return out.to(context.dtype)
