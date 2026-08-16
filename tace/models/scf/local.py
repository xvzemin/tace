################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Iterable, Optional

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

from .electrostatics import build_electrostatics
from .readout import MultipoleReadout
from .utils import (
    compute_total_charge_dipole,
    get_external_field,
    num_graphs,
    require_scf_data,
)


class LocalSourceBase(torch.nn.Module):
    def __init__(self, electrostatics: Optional[Dict] = None) -> None:
        super().__init__()
        self.electrostatics = build_electrostatics(electrostatics)

    def _build_results(
        self,
        density_coefficients: torch.Tensor,
        data: Dict[str, torch.Tensor],
        graph,
        base_results: Dict,
        edge_charge_transfer: Optional[torch.Tensor] = None,
    ) -> Dict:
        n_graphs = num_graphs(data)
        positions = data["positions"] if graph is None else graph.positions
        total_charge, dipole = compute_total_charge_dipole(
            density_coefficients, positions, data["batch"], n_graphs
        )
        electrostatic_energy = self.electrostatics.energy(
            density_coefficients, data, graph
        )
        external_field = get_external_field(data, positions, n_graphs)
        field_energy = torch.sum(dipole * external_field, dim=-1)
        results = dict(base_results)
        results["energy"] = base_results["energy"] + electrostatic_energy + field_energy
        results.update(
            {
                "density_coefficients": density_coefficients,
                "charges": density_coefficients[:, 0],
                "total_charge": total_charge,
                "dipole": dipole,
                "electrostatic_energy": electrostatic_energy,
                "field_energy": field_energy,
                "edge_charge_transfer": edge_charge_transfer,
                "fermi_level": None,
                "scf_residual": None,
                "scf_num_steps": None,
            }
        )
        return results


class LocalCharges(LocalSourceBase):
    """Predict atom-centred multipoles locally without an SCF cycle."""

    def __init__(
        self,
        irreps_in: Iterable[o3.Irreps],
        density_max_l: int = 0,
        electrostatics: Optional[Dict] = None,
    ) -> None:
        super().__init__(electrostatics)
        self.source_readout = MultipoleReadout(irreps_in, density_max_l)

    def forward(self, data, graph, base_results):
        require_scf_data(data)
        density = self.source_readout(base_results["descriptors"])
        return self._build_results(density, data, graph, base_results)


class _FormalCharges(torch.nn.Module):
    def __init__(
        self,
        num_elements: int,
        formal_charges_from_data: bool,
        atomic_formal_charges: Optional[list[float]],
    ) -> None:
        super().__init__()
        self.formal_charges_from_data = bool(formal_charges_from_data)
        if self.formal_charges_from_data:
            self.register_buffer("values", torch.empty(0), persistent=False)
        else:
            if atomic_formal_charges is None:
                raise ValueError(
                    "atomic_formal_charges is required when "
                    "formal_charges_from_data=False"
                )
            if len(atomic_formal_charges) != num_elements:
                raise ValueError("atomic_formal_charges must match num_elements")
            self.register_buffer("values", torch.tensor(atomic_formal_charges))

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.formal_charges_from_data:
            charges = data.get("formal_charges", data.get("charges"))
            if charges is None:
                raise KeyError("formal charge models require formal_charges or charges")
            return charges.reshape(-1)
        return torch.matmul(data["node_attrs"], self.values.to(data["node_attrs"]))


class FixedChargeBaseline(LocalSourceBase):
    """Use fixed formal monopoles with a learned short-range TACE energy."""

    def __init__(
        self,
        num_elements: int,
        formal_charges_from_data: bool = True,
        atomic_formal_charges: Optional[list[float]] = None,
        electrostatics: Optional[Dict] = None,
    ) -> None:
        super().__init__(electrostatics)
        self.formal_charges = _FormalCharges(
            num_elements, formal_charges_from_data, atomic_formal_charges
        )

    def forward(self, data, graph, base_results):
        require_scf_data(data)
        density = self.formal_charges(data).unsqueeze(-1)
        return self._build_results(density, data, graph, base_results)


class LocalSplitCharges(LocalSourceBase):
    """Predict locally conserved edge charge transfer and atomic multipoles."""

    def __init__(
        self,
        irreps_in: Iterable[o3.Irreps],
        scalar_descriptor_dim: int,
        num_elements: int,
        density_max_l: int = 0,
        formal_charges_from_data: bool = True,
        atomic_formal_charges: Optional[list[float]] = None,
        transfer_hidden: int = 64,
        electrostatics: Optional[Dict] = None,
    ) -> None:
        super().__init__(electrostatics)
        self.density_max_l = int(density_max_l)
        self.source_readout = MultipoleReadout(irreps_in, density_max_l)
        self.formal_charges = _FormalCharges(
            num_elements, formal_charges_from_data, atomic_formal_charges
        )
        self.transfer = torch.nn.Sequential(
            torch.nn.Linear(2 * scalar_descriptor_dim + 1, transfer_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(transfer_hidden, 1),
        )

    def forward(self, data, graph, base_results):
        require_scf_data(data)
        density = self.source_readout(base_results["descriptors"])
        density = torch.cat([torch.zeros_like(density[:, :1]), density[:, 1:]], dim=-1)
        density = torch.cat(
            [self.formal_charges(data).unsqueeze(-1), density[:, 1:]], dim=-1
        )

        source, target = data["edge_index"]
        scalar = base_results["scalar_descriptor"]
        edge_length = graph.edge_length if graph is not None else data["edge_length"]
        transfer_input = torch.cat(
            [
                scalar.index_select(0, source),
                scalar.index_select(0, target),
                edge_length,
            ],
            dim=-1,
        )
        edge_transfer = self.transfer(transfer_input).squeeze(-1)
        delta = torch.zeros_like(density[:, 0])
        delta.index_add_(0, target, edge_transfer)
        delta.index_add_(0, source, -edge_transfer)
        density = torch.cat(
            [(density[:, 0] + delta).unsqueeze(-1), density[:, 1:]], dim=-1
        )
        return self._build_results(
            density, data, graph, base_results, edge_charge_transfer=edge_transfer
        )
