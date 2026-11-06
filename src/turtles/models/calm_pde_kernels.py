"""Fused continuous-convolution decode for CALM-PDE.

The decoder evaluates ``Q`` query points (the output mesh) against a small shared
latent cloud. Materialising the per-query neighbour gather ``(B, Q, K, Cin)`` and
the factorised kernel ``(B, Q, Cin, H)`` does not scale to million-point meshes,
so on CUDA a Triton kernel runs one program per ``(batch, query)`` and keeps those
intermediates in SRAM; only the ``(B, Q, Cout)`` output is written.

``_decode_block`` is the single pure-torch definition of the math. It is the
CPU / no-Triton path, the correctness oracle for the kernel, and the function
re-run in chunks inside the custom backward (``_FusedDecode``), which bounds the
backward memory by ``bwd_chunk`` queries regardless of ``Q``.

Shapes: ``query_pos (B, Q, D)``; ``latent_pos (M, D)`` shared across the batch (the
decoder) or ``(B, M, D)`` per sample (the encoder); ``latent_feat (B, M, Cin)``.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from turtles.models.knn import knn

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # triton is optional
    HAS_TRITON = False


def fused_kernel_supported(kernel_mlp: torch.nn.Sequential) -> bool:
    """Whether the Triton kernel can run this kernel MLP.

    ``tl.dot`` needs tile sides >= 16 and ``tl.arange`` needs powers of two, so the
    hidden width must be a power of two of at least 16 (the paper uses 64). Other
    widths transparently use the pure-torch reference path.
    """
    hidden = next(iter(kernel_mlp)).weight.shape[0]
    return hidden >= 16 and hidden & (hidden - 1) == 0


def knn_search_chunked(
    query_pos: torch.Tensor, latent_pos: torch.Tensor, K: int, *, search_chunk: int = 262_144
) -> torch.Tensor:
    """Neighbour indices ``(B, Q, K)`` as int32 (the latent cloud is far below 2^31 points)."""
    B, Q, _ = query_pos.shape
    out = torch.empty((B, Q, K), device=query_pos.device, dtype=torch.int32)
    step = max(1, search_chunk // B)
    for s in range(0, Q, step):
        e = min(s + step, Q)
        out[:, s:e] = knn(query_pos[:, s:e], latent_pos, K).to(torch.int32)
    return out


def mlp_params(kernel_mlp: torch.nn.Sequential):
    """``(W1, b1, W2, b2, W3, b3)`` of ``[Linear, GELU, Linear, GELU, Linear]``."""
    lin1, _, lin2, _, lin3 = list(kernel_mlp)
    return lin1.weight, lin1.bias, lin2.weight, lin2.bias, lin3.weight, lin3.bias


def _decode_block(
    query_pos, latent_pos, latent_feat, idx, w1, b1, w2, b2, w3, b3,
    *, in_channels, out_channels, temperature, eps,
):  # fmt: skip
    """Differentiable decode of a block of queries -> ``(B, Q, Cout)``.

    For each query: gather its ``K`` neighbours, evaluate the kernel MLP on
    ``(delta, |delta|^2)``, weight by ``softmax(-|delta|^2 / T)`` and contract with the
    ``Cin x Cout`` kernel, factorised so the ``(Cin*Cout)`` tensor is never formed.
    """
    Cin, Cout = in_channels, out_channels
    H = w2.size(0)
    B, Q = query_pos.shape[:2]
    idx = idx.long()
    bi = torch.arange(B, device=query_pos.device).view(B, 1, 1)
    sp = latent_pos[bi, idx] if latent_pos.dim() == 3 else latent_pos[idx]  # (B, Q, K, D)
    xk = latent_feat[bi, idx]  # (B, Q, K, Cin)
    delta = query_pos.unsqueeze(2) - sp
    dist2 = (delta * delta).sum(-1)  # (B, Q, K)
    kin = torch.cat([delta, dist2.unsqueeze(-1)], -1)
    h2 = F.gelu(F.linear(F.gelu(F.linear(kin, w1, b1)), w2, b2))  # (B, Q, K, H)
    wgt = torch.softmax(-dist2 / (temperature + eps), -1)
    wx = wgt.unsqueeze(-1) * xk  # (B, Q, K, Cin)
    A = wx.sum(2)  # (B, Q, Cin)
    M = torch.matmul(wx.transpose(-2, -1), h2)  # (B, Q, Cin, H)
    term1 = torch.matmul(A, b3.view(Cin, Cout))
    w3_flat = w3.view(Cin, Cout, H).permute(0, 2, 1).reshape(-1, Cout)
    term2 = torch.matmul(M.reshape(B, Q, -1), w3_flat)
    return term1 + term2


def decode_reference(
    query_pos, latent_pos, latent_feat, knn_idx, kernel_mlp,
    *, in_channels, out_channels, temperature, eps=1e-8, chunk=8192,
):  # fmt: skip
    """Pure-torch decode chunked over queries -> ``(B, Q, Cout)``. Differentiable."""
    w1, b1, w2, b2, w3, b3 = mlp_params(kernel_mlp)
    Q = query_pos.size(1)
    blocks = [
        _decode_block(
            query_pos[:, s : s + chunk],
            latent_pos,
            latent_feat,
            knn_idx[:, s : s + chunk],
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            in_channels=in_channels,
            out_channels=out_channels,
            temperature=temperature,
            eps=eps,
        )
        for s in range(0, Q, chunk)
    ]
    return torch.cat(blocks, 1)


if HAS_TRITON:

    @triton.jit
    def _gelu(x):
        return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))

    @triton.jit
    def _agg_kernel(
        query_ptr, lat_pos_ptr, lat_feat_ptr, idx_ptr, out_ptr,
        w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr,
        BQ, Q, M, inv_temperature,
        D: tl.constexpr, Cin: tl.constexpr, Cout: tl.constexpr,
        H: tl.constexpr, K: tl.constexpr,
        DIN: tl.constexpr, BC: tl.constexpr, COUT_PAD: tl.constexpr, KB: tl.constexpr,
    ):  # fmt: skip
        pid = tl.program_id(0)  # flat (batch, query) index
        if pid >= BQ:
            return
        b = pid // Q
        feat_base = b * M * Cin

        k = tl.arange(0, KB)
        kvalid = k < K
        idx = tl.load(idx_ptr + pid * K + k, mask=kvalid, other=0)

        din = tl.arange(0, DIN)
        col = tl.where(din < D, din, 0)
        sp = tl.load(lat_pos_ptr + idx[:, None] * D + col[None, :])
        qp = tl.load(query_ptr + pid * D + col)
        delta = tl.where(din[None, :] < D, qp[None, :] - sp, 0.0)
        dist2 = tl.sum(delta * delta, axis=1)
        kin = delta + tl.where(din[None, :] == D, dist2[:, None], 0.0)

        hh = tl.arange(0, H)
        w1 = tl.load(w1_ptr + hh[:, None] * DIN + din[None, :])
        h1 = _gelu(tl.dot(kin, tl.trans(w1)) + tl.load(b1_ptr + hh)[None, :])
        w2 = tl.load(w2_ptr + hh[:, None] * H + hh[None, :])
        h2 = _gelu(tl.dot(h1, tl.trans(w2)) + tl.load(b2_ptr + hh)[None, :])

        s = tl.where(kvalid, -dist2 * inv_temperature, float("-inf")).to(tl.float32)
        ex = tl.exp(s - tl.max(s, axis=0))
        wgt = (ex / tl.sum(ex, axis=0)).to(tl.float32)

        co = tl.arange(0, COUT_PAD)
        cmask = co < Cout
        acc = tl.zeros((COUT_PAD,), dtype=tl.float32)
        for c0 in range(0, Cin, BC):
            cidx = c0 + tl.arange(0, BC)
            inb = cidx < Cin
            xk = tl.load(
                lat_feat_ptr + feat_base + idx[:, None] * Cin + cidx[None, :],
                mask=inb[None, :], other=0.0,
            )  # fmt: skip
            wx = (wgt[:, None] * xk).to(tl.float32)
            A = tl.sum(wx, axis=0)
            Mblk = tl.dot(tl.trans(wx), h2)
            b3b = tl.load(
                b3_ptr + cidx[:, None] * Cout + co[None, :],
                mask=inb[:, None] & cmask[None, :], other=0.0,
            )  # fmt: skip
            acc += tl.sum(A[:, None] * b3b, axis=0)
            for cc in range(0, Cout):
                w3b = tl.load(
                    w3_ptr + (cidx[:, None] * Cout + cc) * H + hh[None, :],
                    mask=inb[:, None], other=0.0,
                )  # fmt: skip
                acc += tl.where(co == cc, tl.sum(tl.sum(Mblk * w3b, axis=1), axis=0), 0.0)

        tl.store(out_ptr + pid * Cout + co, acc, mask=cmask)


def fused_aggregate(
    query_pos, latent_pos, latent_feat, idx, kernel_mlp,
    *, in_channels, out_channels, temperature, eps=1e-8,
):  # fmt: skip
    """Triton aggregation -> ``(B, Q, Cout)`` fp32; one program per (batch, query)."""
    w1, b1, w2, b2, w3, b3 = mlp_params(kernel_mlp)
    B, Q, D = query_pos.shape
    M = latent_pos.size(0)
    Cin, Cout = in_channels, out_channels
    H = w2.size(0)
    K = idx.size(2)
    DIN = max(16, triton.next_power_of_2(D + 1))
    COUT_PAD = max(16, triton.next_power_of_2(Cout))
    KB = max(16, triton.next_power_of_2(K))
    out = torch.empty((B, Q, Cout), device=query_pos.device, dtype=torch.float32)
    w1p = torch.zeros((H, DIN), device=w1.device, dtype=torch.float32)
    w1p[:, : D + 1] = w1.float()
    _agg_kernel[(B * Q,)](
        query_pos.float().contiguous(), latent_pos.float().contiguous(),
        latent_feat.float().contiguous(), idx.contiguous(), out,
        w1p.contiguous(), b1.float().contiguous(),
        w2.float().contiguous(), b2.float().contiguous(),
        w3.float().contiguous(), b3.float().contiguous(),
        B * Q, Q, M, float(1.0 / (temperature + eps)),
        D=D, Cin=Cin, Cout=Cout, H=H, K=K, DIN=DIN, BC=64, COUT_PAD=COUT_PAD, KB=KB,
    )  # fmt: skip
    return out


def _autocast_state(device_type: str) -> tuple[bool, torch.dtype | None]:
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:  # torch < 2.4
        enabled = torch.is_autocast_enabled() if device_type == "cuda" else False
    if not enabled:
        return False, None
    try:
        dtype = torch.get_autocast_dtype(device_type)
    except AttributeError:  # torch < 2.4
        dtype = torch.get_autocast_gpu_dtype()
    return True, dtype


class _Linear:
    def __init__(self, weight, bias):
        self.weight, self.bias = weight, bias


class _Sequential:
    """Lets :func:`mlp_params` unpack raw weight tensors."""

    def __init__(self, w1, b1, w2, b2, w3, b3):
        self._items = [_Linear(w1, b1), None, _Linear(w2, b2), None, _Linear(w3, b3)]

    def __iter__(self):
        return iter(self._items)


_compiled_block = None


def _get_compiled_block():
    global _compiled_block
    if _compiled_block is None:
        _compiled_block = torch.compile(_decode_block, dynamic=True)
    return _compiled_block


class _FusedDecode(torch.autograd.Function):
    """Fused forward kernel + chunked eager-recompute backward."""

    @staticmethod
    def forward(ctx, query_pos, latent_pos, latent_feat, idx, w1, b1, w2, b2, w3, b3,
                in_channels, out_channels, temperature, eps, bwd_chunk, compiled):  # fmt: skip
        ctx.save_for_backward(query_pos, latent_pos, latent_feat, idx, w1, b1, w2, b2, w3, b3)
        ctx.cfg = (in_channels, out_channels, temperature, eps, bwd_chunk)
        ctx.compiled = compiled
        ctx.autocast, ctx.autocast_dtype = _autocast_state(query_pos.device.type)
        with torch.no_grad():
            return fused_aggregate(
                query_pos, latent_pos, latent_feat, idx, _Sequential(w1, b1, w2, b2, w3, b3),
                in_channels=in_channels, out_channels=out_channels,
                temperature=temperature, eps=eps,
            )  # fmt: skip

    @staticmethod
    def backward(ctx, grad_out):
        query_pos, latent_pos, latent_feat, idx, w1, b1, w2, b2, w3, b3 = ctx.saved_tensors
        Cin, Cout, T, eps, chunk = ctx.cfg
        block = _get_compiled_block() if ctx.compiled else _decode_block
        need_q, need_lp, need_lf = ctx.needs_input_grad[0:3]
        need_w = ctx.needs_input_grad[4:10]
        params = [w1, b1, w2, b2, w3, b3]

        g_query = torch.zeros_like(query_pos) if need_q else None
        g_latpos = torch.zeros_like(latent_pos) if need_lp else None
        g_latfeat = torch.zeros_like(latent_feat) if need_lf else None
        g_params = [torch.zeros_like(p) if n else None for p, n in zip(params, need_w, strict=True)]

        amp = (
            torch.autocast(query_pos.device.type, dtype=ctx.autocast_dtype)
            if ctx.autocast
            else contextlib.nullcontext()
        )
        Q = query_pos.size(1)
        for s in range(0, Q, chunk):
            e = min(s + chunk, Q)
            qp = query_pos[:, s:e].detach().requires_grad_(need_q)
            lp = latent_pos.detach().requires_grad_(need_lp)
            lf = latent_feat.detach().requires_grad_(need_lf)
            ps = [p.detach().requires_grad_(n) for p, n in zip(params, need_w, strict=True)]
            leaves = [t for t in [qp, lp, lf, *ps] if t.requires_grad]
            with torch.enable_grad(), amp:
                out_c = block(
                    qp, lp, lf, idx[:, s:e], *ps,
                    in_channels=Cin, out_channels=Cout, temperature=T, eps=eps,
                )  # fmt: skip
            grads = iter(torch.autograd.grad(out_c, leaves, grad_out[:, s:e].to(out_c.dtype)))
            if need_q:
                g_query[:, s:e] = next(grads)
            if need_lp:
                g_latpos += next(grads)
            if need_lf:
                g_latfeat += next(grads)
            for i, needed in enumerate(need_w):
                if needed:
                    g_params[i] += next(grads)
        return (g_query, g_latpos, g_latfeat, None, *g_params, None, None, None, None, None, None)


def fused_decode(
    query_pos, latent_pos, latent_feat, kernel_mlp,
    *, in_channels, out_channels, num_neighbors, temperature, eps=1e-8, search_chunk=262_144,
):  # fmt: skip
    """Inference decode (no grad): fused on CUDA with Triton, torch reference otherwise."""
    K = min(num_neighbors, latent_pos.size(0))
    idx = knn_search_chunked(query_pos, latent_pos, K, search_chunk=search_chunk)
    kwargs = dict(
        in_channels=in_channels, out_channels=out_channels, temperature=temperature, eps=eps
    )
    if HAS_TRITON and query_pos.is_cuda and fused_kernel_supported(kernel_mlp):
        return fused_aggregate(query_pos, latent_pos, latent_feat, idx, kernel_mlp, **kwargs)
    return decode_reference(query_pos, latent_pos, latent_feat, idx, kernel_mlp, **kwargs)


def fused_decode_autograd(
    query_pos, latent_pos, latent_feat, kernel_mlp,
    *, in_channels, out_channels, num_neighbors, temperature, eps=1e-8,
    search_chunk=262_144, bwd_chunk=8192, compiled_backward=True,
):  # fmt: skip
    """Differentiable decode: fused forward + chunked backward on CUDA, reference otherwise."""
    K = min(num_neighbors, latent_pos.size(0))
    idx = knn_search_chunked(query_pos, latent_pos, K, search_chunk=search_chunk)
    if HAS_TRITON and query_pos.is_cuda and fused_kernel_supported(kernel_mlp):
        w1, b1, w2, b2, w3, b3 = mlp_params(kernel_mlp)
        return _FusedDecode.apply(
            query_pos, latent_pos, latent_feat, idx, w1, b1, w2, b2, w3, b3,
            in_channels, out_channels, temperature, eps, bwd_chunk, compiled_backward,
        )  # fmt: skip
    return decode_reference(
        query_pos, latent_pos, latent_feat, idx, kernel_mlp,
        in_channels=in_channels, out_channels=out_channels, temperature=temperature, eps=eps,
    )  # fmt: skip
