from __future__ import annotations

import pytest
import torch

from turtles.models import MODEL_NAMES, build_model, count_parameters
from turtles.models.knn import knn
from turtles.models.layers import SincosEmbedding

B, T, N, C = 2, 1, 96, 5


def tiny_config(name: str) -> dict:
    base = {
        "name": name,
        "space_dim": 3,
        "fun_dim": C,
        "out_dim": C,
        "num_steps": T,
        "num_params": 1,
    }
    extra = {
        "calm_pde": dict(
            num_latent_points=27,
            knn_k=4,
            kernel_hidden=16,
            dim=24,
            num_heads=4,
            depth=1,
            coord_min=[-1.0, -0.5, 0.0],
            coord_max=[2.0, 0.5, 6.3],
            use_checkpoint=False,
            decoder_query_chunk_size=40,
        ),
        "upt": dict(
            dim=24,
            num_heads=4,
            perc_num_heads=4,
            num_supernodes=16,
            knn_k=4,
            num_latent_tokens=8,
            enc_depth=1,
            latent_depth=1,
            dec_depth=1,
            use_checkpoint=False,
        ),
        "transolver": dict(n_layers=2, n_hidden=32, n_head=4, slice_num=4, use_checkpoint=False),
        "persistence": {"num_params": 0},
        "mean_field": {"num_params": 0},
    }
    return {**base, **extra.get(name, {})}


def inputs(num_params: int = 1):
    torch.manual_seed(0)
    context = torch.randn(B, T, N, C)
    coords = torch.rand(B, N, 3) * torch.tensor([3.0, 1.0, 6.28]) + torch.tensor([-1.0, -0.5, 0.0])
    params = torch.rand(B, num_params) * 2 - 1 if num_params else None
    return context, coords, params


@pytest.mark.parametrize("name", ["calm_pde", "upt", "transolver"])
def test_learned_models_forward_backward(name):
    model = build_model(tiny_config(name))
    assert count_parameters(model) > 0
    context, coords, params = inputs()
    out = model(context, coords, params)
    assert out.shape == (B, N, C) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("name", ["calm_pde", "upt"])
def test_query_subsampling_and_full_inference(name):
    model = build_model(tiny_config(name)).eval()
    assert model.supports_query_coords
    context, coords, params = inputs()
    with torch.no_grad():
        sub = model(context, coords, params, query_coords=coords[:, :10])
        assert sub.shape == (B, 10, C)
        torch.manual_seed(1)
        full = model(context, coords, params)
        torch.manual_seed(1)
        chunked = model.infer_full(context, coords, params, query_chunk_size=37)
    assert chunked.shape == (B, N, C)
    if name == "calm_pde":  # deterministic: encoder sees the same mesh, decoder chunking is exact
        torch.testing.assert_close(chunked, full, atol=1e-5, rtol=1e-4)


def test_calm_pde_without_bounds_uses_learnable_grid():
    cfg = tiny_config("calm_pde")
    cfg.update(coord_min=None, coord_max=None, learn_latent_grid=True, num_latent_points=20)
    model = build_model(cfg)
    assert model.encoder.latent_grid.requires_grad
    assert model.encoder.latent_grid is model.decoder.latent_grid
    with pytest.raises(ValueError):
        build_model({**tiny_config("calm_pde"), "num_latent_points": 30})  # not a cube


def test_transolver_rejects_query_coords():
    model = build_model(tiny_config("transolver"))
    context, coords, params = inputs()
    with pytest.raises(ValueError):
        model(context, coords, params, query_coords=coords[:, :10])


@pytest.mark.parametrize("name", ["persistence", "mean_field"])
def test_baselines(name):
    model = build_model(tiny_config(name))
    context, coords, _ = inputs(0)
    out = model(context, coords)
    assert out.shape == (B, N, C)
    if name == "persistence":
        torch.testing.assert_close(out, context[:, -1])
    else:
        torch.testing.assert_close(out[:, 0], context[:, -1].mean(dim=1))
        assert model(context, coords, query_coords=coords[:, :7]).shape == (B, 7, C)


def test_registry_lists_all_models():
    assert set(MODEL_NAMES) == {
        "calm_pde",
        "upt",
        "transolver",
        "gino",
        "persistence",
        "mean_field",
    }
    with pytest.raises(KeyError):
        build_model({"name": "nope"})


def test_gino_adapter_if_available():
    pytest.importorskip("neuralop")
    cfg = {
        **tiny_config("gino"),
        "latent_grid_resolution": 6,
        "fno_n_modes": [2, 2, 2],
        "fno_hidden_channels": 8,
        "fno_n_layers": 1,
        "in_gno_radius": 0.5,
        "out_gno_radius": 0.5,
        "gno_use_open3d": False,
    }
    model = build_model(cfg).eval()
    context, coords, params = inputs()
    with torch.no_grad():
        out = model(context[:1], coords[:1], params[:1])
    assert out.shape == (1, N, C) and torch.isfinite(out).all()


def test_knn_torch_fallback_matches_bruteforce():
    torch.manual_seed(0)
    q = torch.randn(2, 50, 3)
    c = torch.randn(2, 200, 3)
    idx, dist2 = knn(q, c, 5, return_distances=True, chunk=7)
    brute = torch.cdist(q, c).topk(5, dim=-1, largest=False)
    assert torch.equal(idx, brute.indices)
    torch.testing.assert_close(dist2, brute.values.square(), atol=1e-5, rtol=1e-4)
    shared = knn(q, c[0], 3)
    assert shared.shape == (2, 50, 3)


def test_sincos_embedding_dim_check():
    emb = SincosEmbedding(24, 3)
    assert emb(torch.rand(4, 7, 3)).shape == (4, 7, 24)
    with pytest.raises(ValueError):
        SincosEmbedding(25, 3)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _decoder():
    from turtles.models.calm_pde import ContinuousConvDecoder

    torch.manual_seed(0)
    dec = ContinuousConvDecoder(12, 5, 3, num_neighbors=8, hidden_channels=16).eval()
    dec.latent_grid = torch.nn.Parameter(torch.rand(64, 3))
    return dec


def test_fused_reference_matches_eager_decoder():
    from turtles.models.calm_pde_kernels import decode_reference, knn_search_chunked

    dec = _decoder()
    dec.use_fused = False
    feat, query = torch.randn(3, 64, 12), torch.rand(3, 200, 3)
    with torch.no_grad():
        eager = dec(feat, query)
        idx = knn_search_chunked(query, dec.latent_grid.detach(), 8)
        ref = (
            decode_reference(
                query,
                dec.latent_grid.detach(),
                feat,
                idx,
                dec.kernel_mlp,
                in_channels=12,
                out_channels=5,
                temperature=dec.temperature,
            )
            + dec.bias
        )
    assert torch.allclose(ref, eager, atol=1e-4)


@cuda
@pytest.mark.cuda
def test_fused_kernel_matches_reference_cuda():
    from turtles.models.calm_pde_kernels import (
        HAS_TRITON,
        decode_reference,
        fused_decode,
        knn_search_chunked,
    )

    if not HAS_TRITON:
        pytest.skip("triton not installed")
    dec = _decoder().cuda()
    feat, query = torch.randn(4, 64, 12, device="cuda"), torch.rand(4, 300, 3, device="cuda")
    lp = dec.latent_grid.detach()
    kw = dict(in_channels=12, out_channels=5, temperature=dec.temperature)
    with torch.no_grad():
        idx = knn_search_chunked(query, lp, 8)
        ref = decode_reference(query, lp, feat, idx, dec.kernel_mlp, **kw)
        fused = fused_decode(query, lp, feat, dec.kernel_mlp, num_neighbors=8, **kw)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fused_amp = fused_decode(query, lp, feat, dec.kernel_mlp, num_neighbors=8, **kw)
    assert fused.shape == (4, 300, 5)
    assert torch.allclose(fused, ref, atol=1e-3)
    assert torch.allclose(fused_amp.float(), ref, atol=1e-2)


@cuda
@pytest.mark.cuda
def test_fused_gradients_match_reference_cuda():
    from turtles.models.calm_pde_kernels import (
        HAS_TRITON,
        decode_reference,
        fused_decode_autograd,
        knn_search_chunked,
    )

    if not HAS_TRITON:
        pytest.skip("triton not installed")
    dec = _decoder().cuda()
    lp = dec.latent_grid.detach()
    query = torch.rand(4, 220, 3, device="cuda")
    feat0 = torch.randn(4, 64, 12, device="cuda")
    grad_out = torch.randn(4, 220, 5, device="cuda")
    kw = dict(in_channels=12, out_channels=5, temperature=dec.temperature)

    def run(fused: bool):
        feat = feat0.clone().requires_grad_(True)
        for p in dec.kernel_mlp.parameters():
            p.grad = None
        if fused:
            out = fused_decode_autograd(query, lp, feat, dec.kernel_mlp, num_neighbors=8, **kw)
        else:
            out = decode_reference(
                query, lp, feat, knn_search_chunked(query, lp, 8), dec.kernel_mlp, **kw
            )
        (out * grad_out).sum().backward()
        return feat.grad.clone(), [p.grad.clone() for p in dec.kernel_mlp.parameters()]

    g_fused, w_fused = run(True)
    g_ref, w_ref = run(False)
    assert torch.allclose(g_fused, g_ref, atol=1e-3)
    assert all(torch.allclose(a, b, atol=1e-3) for a, b in zip(w_fused, w_ref, strict=True))


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("kernel_hidden", [16, 8])  # fused kernel, then the reference fallback
def test_calm_pde_full_model_cuda_forward_backward(kernel_hidden):
    from turtles.models.calm_pde_kernels import fused_kernel_supported

    model = build_model({**tiny_config("calm_pde"), "kernel_hidden": kernel_hidden}).cuda()
    assert fused_kernel_supported(model.decoder.kernel_mlp) == (kernel_hidden == 16)
    context, coords, params = inputs()
    context, coords, params = context.cuda(), coords.cuda(), params.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(context, coords, params)
    assert out.shape == (B, N, C) and torch.isfinite(out).all()
    out.float().square().mean().backward()
    assert model.encoder.kernel_mlp[0].weight.grad is not None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model.infer_full(context, coords, params, query_chunk_size=37)
    assert full.shape == (B, N, C) and torch.isfinite(full).all()


def test_gino_fast_kernels_match_neuralop_native():
    pytest.importorskip("neuralop")
    from neuralop.layers.neighbor_search import native_neighbor_search
    from neuralop.layers.segment_csr import segment_csr as native_segment_csr

    from turtles.models.gino import chunked_radius_search, fast_segment_csr

    torch.manual_seed(0)
    data, queries = torch.rand(500, 3), torch.rand(70, 3)
    ref = native_neighbor_search(data, queries, 0.2)
    ours = chunked_radius_search(data, queries, 0.2, chunk=16)
    assert torch.equal(ours["neighbors_row_splits"], ref["neighbors_row_splits"])
    assert torch.equal(ours["neighbors_index"], ref["neighbors_index"])

    indptr = ref["neighbors_row_splits"]
    src = torch.randn(int(indptr[-1]), 6)
    for reduce in ("sum", "mean"):
        expected = native_segment_csr(src, indptr, reduce=reduce, use_scatter=False)
        torch.testing.assert_close(
            fast_segment_csr(src, indptr, reduce), expected, atol=1e-5, rtol=1e-5
        )
    batched = torch.randn(2, int(indptr[-1]), 6)
    expected = native_segment_csr(batched, indptr.unsqueeze(0), reduce="mean", use_scatter=False)
    torch.testing.assert_close(
        fast_segment_csr(batched, indptr.unsqueeze(0), "mean"), expected, atol=1e-5, rtol=1e-5
    )
