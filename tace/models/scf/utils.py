################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Optional

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum


def natural_irreps(max_l: int, multiplicity: int = 1) -> o3.Irreps:
    if max_l < 0:
        raise ValueError("max_l must be non-negative")
    return o3.Irreps([(multiplicity, o3.Irrep(l, (-1) ** l)) for l in range(max_l + 1)])


def num_graphs(data: Dict[str, torch.Tensor]) -> int:
    if "ptr" in data:
        return int(data["ptr"].numel() - 1)
    if data["batch"].numel() == 0:
        return 0
    return int(data["batch"].max().item() + 1)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = scatter_sum(src, index, dim=0, dim_size=dim_size)
    count = scatter_sum(
        torch.ones_like(src[..., 0] if src.ndim > 1 else src),
        index,
        dim=0,
        dim_size=dim_size,
    )
    while count.ndim < out.ndim:
        count = count.unsqueeze(-1)
    return out / count.clamp_min(1.0)


def enforce_total_charge(
    density_coefficients: torch.Tensor,
    total_charge: torch.Tensor,
    batch: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project monopoles onto the requested per-graph total charge."""
    n_graphs = int(total_charge.numel())
    current = scatter_sum(density_coefficients[:, 0], batch, dim=0, dim_size=n_graphs)
    deficit = total_charge.reshape(-1) - current
    if weights is None:
        weights = torch.ones_like(density_coefficients[:, 0])
    normalization = scatter_sum(weights, batch, dim=0, dim_size=n_graphs)
    counts = scatter_sum(torch.ones_like(weights), batch, dim=0, dim_size=n_graphs)
    uniform = torch.reciprocal(counts.index_select(0, batch).clamp_min(1.0))
    denominator = normalization.index_select(0, batch)
    safe_denominator = torch.where(
        denominator.abs() > 1.0e-12, denominator, torch.ones_like(denominator)
    )
    normalized = torch.where(
        denominator.abs() > 1.0e-12,
        weights / safe_denominator,
        uniform,
    )
    correction = normalized * deficit.index_select(0, batch)
    monopoles = density_coefficients[:, 0] + correction
    return torch.cat([monopoles.unsqueeze(-1), density_coefficients[:, 1:]], dim=-1)


def compute_total_charge_dipole(
    density_coefficients: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    n_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_charge = scatter_sum(
        density_coefficients[:, 0], batch, dim=0, dim_size=n_graphs
    )
    dipole = scatter_sum(
        positions * density_coefficients[:, :1],
        batch,
        dim=0,
        dim_size=n_graphs,
    )
    if density_coefficients.shape[-1] >= 4:
        intrinsic = scatter_sum(
            density_coefficients[:, 1:4], batch, dim=0, dim_size=n_graphs
        )
        dipole = dipole + intrinsic[:, [2, 0, 1]]
    return total_charge, dipole


def get_external_field(
    data: Dict[str, torch.Tensor], reference: torch.Tensor, n_graphs: int
) -> torch.Tensor:
    field = data.get("electric_field", data.get("external_field"))
    if field is None:
        return reference.new_zeros((n_graphs, 3))
    return field.reshape(n_graphs, 3)


def get_total_charge(
    data: Dict[str, torch.Tensor], reference: torch.Tensor, n_graphs: int
) -> torch.Tensor:
    total_charge = data.get("total_charge")
    if total_charge is None:
        return reference.new_zeros(n_graphs)
    return total_charge.reshape(n_graphs).to(reference)


def require_scf_data(data: Dict[str, torch.Tensor]) -> None:
    required = {"positions", "batch"}
    missing = sorted(key for key in required if data.get(key) is None)
    if missing:
        raise KeyError(f"SCF input is missing required keys: {missing}")
