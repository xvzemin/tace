################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Iterable, Optional

import torch
from e3nn import o3

from tace.models.linear import e3nnLinear

from .utils import natural_irreps


class ScalarReadout(torch.nn.Module):
    """Sum scalar readouts from one or more TACE descriptor layers."""

    def __init__(self, irreps_in: Iterable[o3.Irreps], num_outputs: int = 1) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                e3nnLinear(irreps, o3.Irreps(f"{num_outputs}x0e"), bias=True)
                for irreps in irreps_in
            ]
        )

    def forward(self, descriptors: list[torch.Tensor]) -> torch.Tensor:
        values = [layer(x) for layer, x in zip(self.layers, descriptors)]
        if not values:
            raise ValueError("descriptors must contain at least one layer")
        return torch.stack(values, dim=0).sum(dim=0)


class MultipoleReadout(torch.nn.Module):
    """Read natural-parity atom-centred multipoles from TACE descriptors."""

    def __init__(self, irreps_in: Iterable[o3.Irreps], max_l: int) -> None:
        super().__init__()
        self.irreps_out = natural_irreps(max_l)
        self.layers = torch.nn.ModuleList(
            [e3nnLinear(irreps, self.irreps_out, bias=True) for irreps in irreps_in]
        )

    def forward(self, descriptors: list[torch.Tensor]) -> torch.Tensor:
        values = [layer(x) for layer, x in zip(self.layers, descriptors)]
        if not values:
            raise ValueError("descriptors must contain at least one layer")
        return torch.stack(values, dim=0).sum(dim=0)


class DensityResponse(torch.nn.Module):
    """Map electrostatic field features to a geometry-conditioned density update."""

    def __init__(
        self,
        irreps_in: Iterable[o3.Irreps],
        density_max_l: int,
        feature_max_l: int,
        num_feature_widths: int,
    ) -> None:
        super().__init__()
        self.density_irreps = natural_irreps(density_max_l)
        self.feature_irreps = natural_irreps(
            feature_max_l, multiplicity=num_feature_widths
        )
        self.field_linear = e3nnLinear(
            self.feature_irreps, self.density_irreps, bias=False
        )
        self.geometry_gate = ScalarReadout(irreps_in, self.density_irreps.num_irreps)

        repeated_l = []
        for index, (_, irrep) in enumerate(self.density_irreps):
            repeated_l.extend([index] * irrep.dim)
        self.register_buffer(
            "gate_index", torch.tensor(repeated_l, dtype=torch.long), persistent=False
        )

    def forward(
        self, descriptors: list[torch.Tensor], field_features: torch.Tensor
    ) -> torch.Tensor:
        response = self.field_linear(field_features)
        gates = torch.sigmoid(self.geometry_gate(descriptors))
        gates = gates.index_select(-1, self.gate_index)
        return response * gates


class DensityEnergyReadout(torch.nn.Module):
    """Density-dependent local energy used after the SCF solve."""

    MODES = {"none", "linear", "quadratic", "many_body", "many_body_field"}

    def __init__(
        self,
        irreps_in: Iterable[o3.Irreps],
        density_dim: int,
        mode: str = "many_body",
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"density energy mode must be one of {sorted(self.MODES)}")
        self.mode = mode
        self.geometry = ScalarReadout(irreps_in, hidden)
        in_dim = density_dim
        if mode == "many_body_field":
            in_dim += density_dim
        if mode in {"many_body", "many_body_field"}:
            self.network = torch.nn.Sequential(
                torch.nn.Linear(hidden + in_dim, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, 1),
            )
        elif mode == "linear":
            self.network = torch.nn.Linear(density_dim, 1, bias=False)
        elif mode == "quadratic":
            self.linear = torch.nn.Parameter(torch.zeros(density_dim))
            self.log_hardness = torch.nn.Parameter(torch.zeros(density_dim))

    def forward(
        self,
        descriptors: list[torch.Tensor],
        density_coefficients: torch.Tensor,
        field_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "none":
            return density_coefficients.new_zeros(density_coefficients.shape[0])
        if self.mode == "linear":
            return self.network(density_coefficients).squeeze(-1)
        if self.mode == "quadratic":
            hardness = torch.nn.functional.softplus(self.log_hardness)
            return torch.sum(
                self.linear * density_coefficients
                + 0.5 * hardness * density_coefficients.square(),
                dim=-1,
            )
        geometry = self.geometry(descriptors)
        values = [geometry, density_coefficients]
        if self.mode == "many_body_field":
            if field_features is None:
                raise ValueError("many_body_field readout requires field_features")
            if field_features.shape[-1] != density_coefficients.shape[-1]:
                raise ValueError(
                    "many_body_field requires field and density dimensions to match"
                )
            values.append(field_features)
        return self.network(torch.cat(values, dim=-1)).squeeze(-1)
