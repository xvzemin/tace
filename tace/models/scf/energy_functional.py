################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Iterable, Optional

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

from .electrostatics import build_electrostatics
from .readout import DensityEnergyReadout, MultipoleReadout
from .state import EnergyFunctionalSCFOptions
from .utils import (
    compute_total_charge_dipole,
    enforce_total_charge,
    get_external_field,
    get_total_charge,
    num_graphs,
    scatter_mean,
)


class EnergyFunctionalSCF(torch.nn.Module):
    """Variational SCF model that minimizes a learned density functional."""

    def __init__(
        self,
        irreps_in: Iterable[o3.Irreps],
        density_max_l: int = 0,
        electrostatics: Optional[Dict] = None,
        options: Optional[Dict] = None,
        density_energy_mode: str = "many_body",
        density_energy_hidden: int = 64,
    ) -> None:
        super().__init__()
        if density_energy_mode == "none":
            raise ValueError("EnergyFunctionalSCF requires a density energy readout")
        self.options = EnergyFunctionalSCFOptions(
            **({} if options is None else options)
        )
        self.density_dim = (density_max_l + 1) ** 2
        self.local_source = MultipoleReadout(irreps_in, density_max_l)
        self.density_energy = DensityEnergyReadout(
            irreps_in,
            density_dim=self.density_dim,
            mode=density_energy_mode,
            hidden=density_energy_hidden,
        )
        electrostatics = {} if electrostatics is None else dict(electrostatics)
        if electrostatics.get("backend", "graph_longrange") == "graph_longrange":
            electrostatics.setdefault("density_max_l", density_max_l)
            electrostatics.setdefault("feature_max_l", 0)
        self.electrostatics = build_electrostatics(electrostatics)

    def _initial_density(self, data, local_density):
        if self.options.initial_density == "local_guess":
            return local_density
        if self.options.initial_density == "zeros":
            return torch.zeros_like(local_density)
        density = data.get("density_coefficients", data.get("atomic_multipoles"))
        if density is None:
            raise KeyError(
                "initial_density='from_data' requires density_coefficients or "
                "atomic_multipoles"
            )
        return density.to(local_density)

    def _functional(self, density, descriptors, data, graph):
        n_graphs = num_graphs(data)
        node_energy = self.density_energy(descriptors, density)
        local_energy = scatter_sum(node_energy, data["batch"], dim=0, dim_size=n_graphs)
        electrostatic_energy = self.electrostatics.energy(density, data, graph)
        positions = data["positions"] if graph is None else graph.positions
        _, dipole = compute_total_charge_dipole(
            density, positions, data["batch"], n_graphs
        )
        field = get_external_field(data, density, n_graphs)
        field_energy = torch.sum(dipole * field, dim=-1)
        return local_energy + electrostatic_energy + field_energy

    def _project_gradient(self, gradient, data):
        projected = gradient.clone()
        n_graphs = num_graphs(data)
        mean_monopole = scatter_mean(
            projected[:, 0], data["batch"], n_graphs
        ).index_select(0, data["batch"])
        projected = torch.cat(
            [(projected[:, 0] - mean_monopole).unsqueeze(-1), projected[:, 1:]],
            dim=-1,
        )
        return projected

    def _solve(self, local_density, descriptors, data, graph):
        density = self._initial_density(data, local_density)
        n_graphs = num_graphs(data)
        target = get_total_charge(data, density, n_graphs)
        density = enforce_total_charge(density, target, data["batch"])
        history = [density.detach()]
        residual = density.new_full((n_graphs,), float("inf"))
        status = "max_steps_reached"
        num_steps = 0
        keep_steps = self.options.num_autograd_steps
        if keep_steps is None:
            keep_steps = self.options.num_scf_steps
        for step in range(self.options.num_scf_steps):
            density = density.requires_grad_(True)
            energy = self._functional(density, descriptors, data, graph).sum()
            gradient = torch.autograd.grad(energy, density, create_graph=self.training)[
                0
            ]
            gradient = self._project_gradient(gradient, data)
            updated = density - self.options.learning_rate * gradient
            updated = enforce_total_charge(updated, target, data["batch"])
            residual = scatter_mean(
                gradient.abs().mean(dim=-1), data["batch"], n_graphs
            )
            density = updated
            history.append(density.detach())
            num_steps = step + 1
            if bool(torch.all(residual < self.options.scf_tolerance)):
                status = "converged"
                break
            if not torch.all(torch.isfinite(density)):
                status = "diverged"
                break
            if step < self.options.num_scf_steps - keep_steps:
                density = density.detach()
        return density, torch.stack(history, dim=-1), residual, num_steps, status

    def forward(self, data, graph, base_results):
        descriptors = base_results["descriptors"]
        local_density = self.local_source(descriptors)
        density, history, residual, steps, status = self._solve(
            local_density, descriptors, data, graph
        )
        functional_energy = self._functional(density, descriptors, data, graph)
        n_graphs = num_graphs(data)
        positions = data["positions"] if graph is None else graph.positions
        total_charge, dipole = compute_total_charge_dipole(
            density, positions, data["batch"], n_graphs
        )
        results = dict(base_results)
        results["energy"] = base_results["energy"] + functional_energy
        status_code = {
            "converged": 0,
            "max_steps_reached": 1,
            "diverged": 2,
        }[status]
        results.update(
            {
                "density_coefficients": density,
                "charges": density[:, 0],
                "total_charge": total_charge,
                "dipole": dipole,
                "energy_functional": functional_energy,
                "density_history": history,
                "fermi_level": None,
                "scf_residual": residual,
                "scf_num_steps": density.new_full(
                    (n_graphs,), steps, dtype=torch.int64
                ),
                "scf_status": density.new_full(
                    (n_graphs,), status_code, dtype=torch.int64
                ),
            }
        )
        return results
