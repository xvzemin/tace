################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Iterable, Optional

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

from .electrostatics import build_electrostatics
from .readout import ScalarReadout
from .utils import (
    compute_total_charge_dipole,
    get_external_field,
    get_total_charge,
    num_graphs,
    require_scf_data,
)


class QEq(torch.nn.Module):
    """Charge equilibration coupled to a TACE short-range backbone."""

    def __init__(
        self,
        irreps_in: Iterable[o3.Irreps],
        electrostatics: Optional[Dict] = None,
        train_hardness: bool = True,
        read_electronegativity: bool = False,
        read_hardness: bool = False,
        default_hardness: float = 2.0,
        hardness_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.electrostatics = build_electrostatics(electrostatics)
        self.electronegativity_readout = ScalarReadout(irreps_in)
        self.hardness_readout = ScalarReadout(irreps_in)
        self.train_hardness = bool(train_hardness)
        self.read_electronegativity = bool(read_electronegativity)
        self.read_hardness = bool(read_hardness)
        self.default_hardness = float(default_hardness)
        self.hardness_epsilon = float(hardness_epsilon)

    def _coulomb_hessian(self, data, graph, reference):
        probe = torch.zeros_like(reference, requires_grad=True)

        def energy(charges):
            return self.electrostatics.energy(charges.unsqueeze(-1), data, graph).sum()

        return torch.autograd.functional.hessian(
            energy, probe, create_graph=self.training, vectorize=True
        )

    @staticmethod
    def _solve(hessian, electronegativity, hardness, total_charge, batch):
        charges = torch.empty_like(electronegativity)
        for graph_index in range(total_charge.numel()):
            indices = torch.where(batch == graph_index)[0]
            n_atoms = indices.numel()
            matrix = hessian.index_select(0, indices).index_select(1, indices)
            matrix = matrix + torch.diag(hardness.index_select(0, indices))
            kkt = matrix.new_ones((n_atoms + 1, n_atoms + 1))
            kkt[:n_atoms, :n_atoms] = matrix
            kkt[-1, -1] = 0.0
            rhs = matrix.new_empty(n_atoms + 1)
            rhs[:n_atoms] = -electronegativity.index_select(0, indices)
            rhs[-1] = total_charge[graph_index]
            solution = torch.linalg.solve(kkt, rhs)
            charges.index_copy_(0, indices, solution[:-1])
        return charges

    def forward(self, data, graph, base_results):
        require_scf_data(data)
        descriptors = base_results["descriptors"]
        reference = data["positions"][:, 0]
        if self.read_electronegativity:
            electronegativity = data["electronegativity"].reshape(-1).to(reference)
        else:
            electronegativity = self.electronegativity_readout(descriptors)[:, 0]
        if self.read_hardness:
            hardness = data["hardness"].reshape(-1).to(reference)
        else:
            hardness = reference.new_full(reference.shape, self.default_hardness)
        if self.train_hardness:
            hardness = hardness + torch.nn.functional.softplus(
                self.hardness_readout(descriptors)[:, 0]
            )
        hardness = hardness.clamp_min(self.hardness_epsilon)

        n_graphs = num_graphs(data)
        external_field = get_external_field(data, reference, n_graphs)
        electronegativity = electronegativity + torch.sum(
            data["positions"] * external_field.index_select(0, data["batch"]), dim=-1
        )
        total_charge = get_total_charge(data, reference, n_graphs)
        hessian = self._coulomb_hessian(data, graph, reference)
        charges = self._solve(
            hessian, electronegativity, hardness, total_charge, data["batch"]
        )
        density = charges.unsqueeze(-1)
        electrostatic_energy = self.electrostatics.energy(density, data, graph)
        site_energy = scatter_sum(
            electronegativity * charges + 0.5 * hardness * charges.square(),
            data["batch"],
            dim=0,
            dim_size=n_graphs,
        )
        qeq_energy = site_energy + electrostatic_energy
        _, dipole = compute_total_charge_dipole(
            density, data["positions"], data["batch"], n_graphs
        )
        results = dict(base_results)
        results["energy"] = base_results["energy"] + qeq_energy
        results.update(
            {
                "charges": charges,
                "density_coefficients": density,
                "total_charge": total_charge,
                "dipole": dipole,
                "electronegativity": electronegativity,
                "hardness": hardness,
                "qeq_energy": qeq_energy,
                "electrostatic_energy": electrostatic_energy,
                "fermi_level": None,
                "scf_residual": None,
                "scf_num_steps": None,
            }
        )
        return results
