"""Single-frame full-mesh prediction dispatch.

Encoder/decoder models expose ``infer_full`` (encode the full mesh once, decode
query chunks). Monolithic models are run on disjoint tiles of ``query_chunk_size``
nodes, which bounds memory at the price of cross-tile context (the only option when
input and output cardinalities are tied, as for Transolver++).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def tile_partition_predict(
    model: torch.nn.Module,
    context: torch.Tensor,
    coords: torch.Tensor,
    params: torch.Tensor | None,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Run ``model`` independently on consecutive node tiles of at most ``chunk_size``."""
    B, _, N, C = context.shape
    out_dim = getattr(getattr(model, "c", None), "out_dim", C)
    out = torch.empty(B, N, out_dim, device=context.device, dtype=context.dtype)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        out[:, start:end] = model(
            context[:, :, start:end].contiguous(), coords[:, start:end].contiguous(), params
        )
    return out


def supports_full_inference(model: torch.nn.Module) -> bool:
    """Whether ``model`` can encode once and decode in chunks."""
    return callable(getattr(model, "infer_full", None)) and bool(
        getattr(model, "supports_infer_full", True)
    )


@torch.no_grad()
def predict_frame(
    model: torch.nn.Module,
    context: torch.Tensor,
    coords: torch.Tensor,
    params: torch.Tensor | None,
    *,
    query_chunk_size: int,
    input_subset_size: int | None = None,
) -> torch.Tensor:
    """Predict one frame at every node of ``coords``: ``(B, T, N, C) -> (B, N, C)``."""
    if supports_full_inference(model):
        return model.infer_full(
            context,
            coords,
            params,
            input_subset_size=input_subset_size,
            query_chunk_size=query_chunk_size,
        )
    return tile_partition_predict(model, context, coords, params, chunk_size=query_chunk_size)
