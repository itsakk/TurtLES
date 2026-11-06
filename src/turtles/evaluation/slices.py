"""Mesh -> regular 2-D grid projection of a z-slice, for qualitative plots.

Built once per trajectory: nodes within a thin band around the mid-span plane are
selected, a regular ``(H, W)`` grid covers their ``(x, y)`` bounding box, and each
grid cell stores its ``k`` nearest band nodes with inverse-distance weights. Applying
the projection to a field is then one gather and a weighted sum.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from turtles.models.knn import knn


@dataclass(frozen=True)
class GridProjection:
    """Cached projection of one mesh onto a 2-D grid (all tensors on CPU)."""

    band_indices: torch.Tensor  # (M,)
    knn_idx: torch.Tensor  # (H, W, k) into the band
    weights: torch.Tensor  # (H, W, k)
    extent: tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
    z0: float


def build_grid_projection(
    coords: torch.Tensor,
    *,
    grid_size: tuple[int, int] = (256, 256),
    half_thickness: float | None = None,
    knn_k: int = 8,
    idw_power: float = 2.0,
    eps: float = 1e-6,
) -> GridProjection:
    """Create the projection for ``(N, 3)`` coordinates (slice through the median ``z``)."""
    if coords.dim() == 3:
        coords = coords[0]
    xyz = coords.detach().float()
    z = xyz[:, 2]
    z0 = float(z.median())
    z_span = float(z.max() - z.min())
    if half_thickness is None:
        half_thickness = max(0.05 * z_span, eps)
    band = (z - z0).abs() <= half_thickness
    if int(band.sum()) < knn_k:
        band = torch.ones_like(band)
    band_indices = torch.nonzero(band, as_tuple=False).squeeze(-1)
    xy = xyz[band_indices, :2]

    lo, hi = xy.min(dim=0).values, xy.max(dim=0).values
    pad = (hi - lo) * 0.02
    lo, hi = lo - pad, hi + pad
    H, W = grid_size
    ys = torch.linspace(float(lo[1]), float(hi[1]), H)
    xs = torch.linspace(float(lo[0]), float(hi[0]), W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx, dist2 = knn(grid.to(device), xy.unsqueeze(0).to(device), knn_k, return_distances=True)
    inv = 1.0 / dist2.clamp_min(eps).pow(idw_power / 2.0)
    weights = (inv / inv.sum(dim=-1, keepdim=True)).reshape(H, W, -1).cpu()
    return GridProjection(
        band_indices=band_indices.cpu(),
        knn_idx=idx.reshape(H, W, -1).cpu(),
        weights=weights,
        extent=(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])),
        z0=z0,
    )


@torch.no_grad()
def apply_grid_projection(field: torch.Tensor, proj: GridProjection) -> torch.Tensor:
    """Project a per-node field ``(N, C)`` onto the grid: ``(H, W, C)``."""
    band = field[proj.band_indices.to(field.device)]
    H, W, k = proj.knn_idx.shape
    gathered = band[proj.knn_idx.to(field.device).reshape(-1)].reshape(H, W, k, -1)
    weights = proj.weights.to(field.device, field.dtype).unsqueeze(-1)
    return (gathered * weights).sum(dim=2)
