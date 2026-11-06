"""k-nearest-neighbour search shared by the point-cloud models.

On CUDA the fused ``flashlib`` kernel is used when installed (memory-flat, handles
million-point corpora in milliseconds). Elsewhere a chunked ``torch.cdist`` +
``topk`` fallback keeps memory bounded; it is exact but slow for very large
corpora, so install ``turtles[gpu]`` for training and evaluation.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - exercised on GPU machines only
    from flashlib.primitives.knn import flash_knn as _flash_knn
except Exception:  # any import problem means "not available"
    _flash_knn = None

HAS_FLASHLIB = _flash_knn is not None

# Upper bound on the number of entries of one (queries x corpus) distance block.
_MAX_BLOCK_ELEMENTS = 2**27


def knn(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    k: int,
    *,
    return_distances: bool = False,
    chunk: int = 65_536,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Indices (and optionally squared distances) of the ``k`` nearest corpus points.

    Args:
        queries: ``(B, Q, D)``.
        corpus: ``(B, M, D)`` or a shared ``(M, D)`` cloud.
        k: Neighbours per query (clamped to ``M``).
        return_distances: Also return squared Euclidean distances (fp32).
        chunk: Queries per block in the torch fallback.

    Returns:
        ``idx`` ``(B, Q, k)`` int64, or ``(idx, dist2)`` with ``dist2`` ``(B, Q, k)``.
    """
    if queries.dim() != 3:
        raise ValueError(f"queries must be (B, Q, D), got {tuple(queries.shape)}")
    B = queries.shape[0]
    corpus_b = corpus.unsqueeze(0).expand(B, -1, -1) if corpus.dim() == 2 else corpus
    k = min(int(k), corpus_b.shape[1])
    q32 = queries.detach().float().contiguous()
    c32 = corpus_b.detach().float().contiguous()

    if HAS_FLASHLIB and q32.is_cuda:
        out = _flash_knn(q32, c32, k, return_distances=return_distances)
        if return_distances:
            dist2, idx = out
            return idx.long(), dist2.float()
        return out.long()
    return _knn_torch(q32, c32, k, return_distances=return_distances, chunk=chunk)


def _knn_torch(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    k: int,
    *,
    return_distances: bool,
    chunk: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    B, Q, _ = queries.shape
    M = corpus.shape[1]
    step = max(1, min(int(chunk), _MAX_BLOCK_ELEMENTS // max(M, 1)))
    idx = torch.empty((B, Q, k), dtype=torch.long, device=queries.device)
    dist2 = torch.empty((B, Q, k), dtype=torch.float32, device=queries.device)
    for start in range(0, Q, step):
        end = min(start + step, Q)
        block = torch.cdist(queries[:, start:end], corpus)  # (B, q, M)
        values, indices = block.topk(k, dim=-1, largest=False)
        idx[:, start:end] = indices
        dist2[:, start:end] = values.square()
    if return_distances:
        return idx, dist2
    return idx
