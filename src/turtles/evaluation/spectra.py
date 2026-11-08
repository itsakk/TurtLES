"""Spanwise energy spectra (paper, Section 4).

The mesh is a structured extrusion along the periodic spanwise direction
``z ∈ [0, 2π]``: every ``(x, y)`` location carries a ring of 121 uniformly spaced
nodes. The spectrum therefore needs no interpolation: a real FFT along each ring,
``|F|^2`` averaged over the rings, gives ``P_t(k_z)`` at integer wavenumbers
``k_z = 0 .. 60``. Band errors are RMS log-power discrepancies over wavenumber sets
(low ``{1,2,3}``, mid ``{4..15}``, high ``{16..60}`` in the paper).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

BandSpec = Mapping[str, tuple[int, int]]

DEFAULT_BANDS: dict[str, tuple[int, int]] = {"low": (1, 3), "mid": (4, 15), "high": (16, 60)}


@dataclass(frozen=True)
class SpanwiseRings:
    """Ring index map of one mesh.

    ``ring_idx[r, l]`` is the mesh index of the ``l``-th node (sorted by ``z``) of ring
    ``r``; ``wavenumbers`` are the integer modes ``0 .. Lz // 2`` of the ``rfft``.
    """

    ring_idx: torch.Tensor
    wavenumbers: torch.Tensor
    length: int
    num_rings: int


def build_spanwise_rings(
    coords: torch.Tensor | np.ndarray, *, xy_tol: float = 1e-4
) -> SpanwiseRings:
    """Group nodes sharing ``(x, y)`` (within ``xy_tol``) into z-ordered rings.

    Only rings of the modal length are kept (near-wall rings of a different length
    would need padding and bias the spectrum).
    """
    xyz = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
    if xyz.ndim == 3:
        if xyz.shape[0] != 1:
            raise ValueError("expected a single mesh (B = 1)")
        xyz = xyz[0]
    if xyz.shape[-1] != 3:
        raise ValueError("spanwise spectra need 3-D coordinates")
    xy = np.round(xyz[:, :2] / xy_tol).astype(np.int64)
    y_min = xy[:, 1].min()
    y_range = xy[:, 1].max() - y_min + 1
    keys = xy[:, 0] * y_range + (xy[:, 1] - y_min)
    order = np.lexsort((xyz[:, 2].astype(np.float64), keys))
    sorted_keys = keys[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_keys)) + 1])
    ends = np.concatenate([starts[1:], [len(order)]])
    lengths = ends - starts
    length = int(np.bincount(lengths).argmax())
    keep = lengths == length
    if length < 2 or not keep.any():
        raise RuntimeError("could not find spanwise rings; check xy_tol")
    ring_idx = np.stack([order[s:e] for s, e in zip(starts[keep], ends[keep], strict=True)])
    wavenumbers = np.arange(length // 2 + 1, dtype=np.float32)
    return SpanwiseRings(
        ring_idx=torch.from_numpy(ring_idx),
        wavenumbers=torch.from_numpy(wavenumbers),
        length=length,
        num_rings=int(keep.sum()),
    )


@torch.no_grad()
def spanwise_power(field: torch.Tensor, rings: SpanwiseRings) -> torch.Tensor:
    """Power spectrum along ``z`` averaged over rings: ``(N, C) -> (C, K)`` or ``(N,) -> (K,)``."""
    squeeze = field.dim() == 1
    values = field.unsqueeze(-1) if squeeze else field
    idx = rings.ring_idx.to(values.device)
    ring_values = values[idx.reshape(-1)].reshape(rings.num_rings, rings.length, -1).float()
    spectrum = torch.fft.rfft(ring_values, dim=1, norm="ortho")  # (R, K, C)
    power = spectrum.abs().square().mean(dim=0).transpose(0, 1).contiguous()  # (C, K)
    return power[0] if squeeze else power


def band_masks(
    wavenumbers: torch.Tensor, bands: BandSpec = DEFAULT_BANDS
) -> dict[str, torch.Tensor]:
    """Boolean masks over ``wavenumbers`` for inclusive integer ranges."""
    k = wavenumbers
    return {name: (k >= lo) & (k <= hi) for name, (lo, hi) in bands.items()}


@torch.no_grad()
def band_log_rmse(
    power_pred: torch.Tensor,
    power_ref: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    eps: float = 1e-12,
) -> dict[str, float]:
    """``sqrt(mean_{k in band} (log P_pred(k) - log P_ref(k))^2)`` for each band."""
    disc = (power_pred.double() + eps).log() - (power_ref.double() + eps).log()
    out: dict[str, float] = {}
    for name, mask in masks.items():
        mask = mask.to(disc.device)
        out[name] = float(disc[mask].square().mean().sqrt()) if int(mask.sum()) else float("nan")
    return out
