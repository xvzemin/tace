################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Iterable, Optional

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

from .electrostatics import build_electrostatics
from .readout import DensityEnergyReadout, DensityResponse, MultipoleReadout
from .state import FixedPointSCFOptions, SCFState
from .utils import (
    compute_total_charge_dipole,
    enforce_total_charge,
    get_external_field,
    get_total_charge,
    num_graphs,
    scatter_mean,
)


class FixedPointSCF(torch.nn.Module):
    """Kohn--Sham-like learned fixed-point iteration over atomic multipoles."""

    def __init__(
        self,
        irreps_in: Iterable[o3.Irreps],
        density_max_l: int = 0,
        feature_max_l: int = 0,
        feature_smearing_widths: Optional[list[float]] = None,
        electrostatics: Optional[Dict] = None,
        options: Optional[Dict] = None,
        training_mode: str = "unrolled",
        density_energy_mode: str = "many_body",
        density_energy_hidden: int = 64,
    ) -> None:
        super().__init__()
        if feature_smearing_widths is None:
            feature_smearing_widths = [1.0]
        if training_mode not in {"direct", "unrolled", "implicit", "linearize_solve"}:
            raise ValueError(
                "training_mode must be direct, unrolled, implicit, or linearize_solve"
            )
        self.options = FixedPointSCFOptions(**({} if options is None else options))
        self.training_mode = training_mode
        self.density_max_l = int(density_max_l)
        self.density_dim = (density_max_l + 1) ** 2
        self.feature_dim = len(feature_smearing_widths) * (feature_max_l + 1) ** 2
        self.local_source = MultipoleReadout(irreps_in, density_max_l)
        self.response = DensityResponse(
            irreps_in,
            density_max_l=density_max_l,
            feature_max_l=feature_max_l,
            num_feature_widths=len(feature_smearing_widths),
        )
        self.density_energy = DensityEnergyReadout(
            irreps_in,
            density_dim=self.density_dim,
            mode=density_energy_mode,
            hidden=density_energy_hidden,
        )
        electrostatics = {} if electrostatics is None else dict(electrostatics)
        if electrostatics.get("backend", "graph_longrange") == "graph_longrange":
            electrostatics.setdefault("density_max_l", density_max_l)
            electrostatics.setdefault("feature_max_l", feature_max_l)
            electrostatics.setdefault(
                "feature_smearing_widths", feature_smearing_widths
            )
        self.electrostatics = build_electrostatics(electrostatics)

    def _initial_density(self, data, local_density):
        mode = self.options.initial_density
        if mode == "local_guess":
            return local_density
        if mode == "zeros":
            return torch.zeros_like(local_density)
        density = data.get("density_coefficients", data.get("atomic_multipoles"))
        if density is None:
            raise KeyError(
                "initial_density='from_data' requires density_coefficients or "
                "atomic_multipoles"
            )
        return density.to(local_density)

    def _initial_fermi_level(self, data, reference, n_graphs):
        if self.options.initial_fermi_level == "zeros":
            return reference.new_zeros(n_graphs)
        value = data.get("fermi_level")
        if value is None:
            return reference.new_zeros(n_graphs)
        return value.reshape(n_graphs).to(reference)

    def _field_features(self, density, fermi_level, data, graph, node_fermi_level=None):
        features = self.electrostatics.features(density, data, graph)
        node_fermi = node_fermi_level
        if node_fermi is None:
            node_fermi = fermi_level.index_select(0, data["batch"])
        first = features[:, :1] + node_fermi.unsqueeze(-1)
        return torch.cat([first, features[:, 1:]], dim=-1)

    def _fixed_point_map(
        self,
        density,
        fermi_level,
        local_density,
        descriptors,
        data,
        graph,
    ):
        node_fermi = fermi_level.index_select(0, data["batch"])
        if torch.is_grad_enabled():
            node_fermi = node_fermi.requires_grad_(True)
        features = self._field_features(
            density, fermi_level, data, graph, node_fermi_level=node_fermi
        )
        updated = local_density + self.response(descriptors, features)
        if not self.options.constant_charge:
            return updated, fermi_level, features

        n_graphs = num_graphs(data)
        target = get_total_charge(data, density, n_graphs)
        if torch.is_grad_enabled() and updated[:, 0].requires_grad:
            positions = data["positions"] if graph is None else graph.positions
            weights = torch.autograd.grad(
                updated[:, 0].sum(),
                node_fermi,
                retain_graph=True,
                create_graph=self.training or positions.requires_grad,
            )[0]
        else:
            weights = torch.ones_like(updated[:, 0])
        current = scatter_sum(updated[:, 0], data["batch"], dim=0, dim_size=n_graphs)
        response = scatter_sum(weights, data["batch"], dim=0, dim_size=n_graphs)
        safe_response = torch.where(
            response.abs() > 1.0e-12, response, torch.ones_like(response)
        )
        fermi_level = fermi_level + (target - current) / safe_response
        updated = enforce_total_charge(updated, target, data["batch"], weights)
        return updated, fermi_level, features

    def _solve(self, local_density, descriptors, data, graph):
        n_graphs = num_graphs(data)
        density = self._initial_density(data, local_density)
        fermi = self._initial_fermi_level(data, density, n_graphs)
        if self.options.constant_charge:
            target = get_total_charge(data, density, n_graphs)
            density = enforce_total_charge(density, target, data["batch"])
        history = [density.detach()]
        residual = density.new_full((n_graphs,), float("inf"))
        status = "max_steps_reached"
        field_features = None
        num_steps = 0
        keep_steps = self.options.num_autograd_steps
        if keep_steps is None:
            keep_steps = self.options.num_scf_steps
        for step in range(self.options.num_scf_steps):
            updated, updated_fermi, field_features = self._fixed_point_map(
                density, fermi, local_density, descriptors, data, graph
            )
            delta = updated - density
            density = density + self.options.mixing_parameter * delta
            fermi = fermi + self.options.mixing_parameter * (updated_fermi - fermi)
            if self.options.constant_charge:
                density = enforce_total_charge(density, target, data["batch"])
            residual = scatter_mean(delta.abs().mean(dim=-1), data["batch"], n_graphs)
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
                fermi = fermi.detach()
        return SCFState(
            density_coefficients=density,
            fermi_level=fermi,
            field_features=field_features,
            density_history=torch.stack(history, dim=-1),
            num_steps=num_steps,
            status=status,
            residual=residual,
        )

    def _direct(self, local_density, descriptors, data, graph):
        density = data.get("density_coefficients", data.get("atomic_multipoles"))
        if density is None:
            raise KeyError(
                "direct fixed-point training requires density_coefficients or "
                "atomic_multipoles"
            )
        density = density.to(local_density)
        n_graphs = num_graphs(data)
        fermi = self._initial_fermi_level(data, density, n_graphs)
        updated, fermi, features = self._fixed_point_map(
            density, fermi, local_density, descriptors, data, graph
        )
        residual = scatter_mean(
            (updated - density).abs().mean(dim=-1), data["batch"], n_graphs
        )
        return SCFState(
            density_coefficients=updated,
            fermi_level=fermi,
            field_features=features,
            density_history=torch.stack([density.detach(), updated.detach()], dim=-1),
            num_steps=1,
            status="direct",
            residual=residual,
        )

    def _attach_implicit_gradient(self, state, local_density, descriptors, data, graph):
        solved = state.density_coefficients.detach().requires_grad_(True)
        fermi = state.fermi_level.detach()

        def residual_fn(value):
            mapped, _, _ = self._fixed_point_map(
                value, fermi, local_density, descriptors, data, graph
            )
            return (value - mapped).reshape(-1)

        jacobian = torch.autograd.functional.jacobian(
            residual_fn, solved, create_graph=False, vectorize=True
        ).reshape(solved.numel(), solved.numel())
        residual = residual_fn(solved)
        correction = torch.linalg.solve(jacobian.detach(), -residual)
        correction = correction.reshape_as(solved)
        density = solved.detach() + correction - correction.detach()
        return state._replace(density_coefficients=density)

    def _build_results(self, state, data, graph, base_results):
        density = state.density_coefficients
        n_graphs = num_graphs(data)
        positions = data["positions"] if graph is None else graph.positions
        total_charge, dipole = compute_total_charge_dipole(
            density, positions, data["batch"], n_graphs
        )
        electrostatic_energy = self.electrostatics.energy(density, data, graph)
        external_field = get_external_field(data, density, n_graphs)
        field_energy = torch.sum(dipole * external_field, dim=-1)
        density_node_energy = self.density_energy(
            base_results["descriptors"], density, state.field_features
        )
        density_energy = scatter_sum(
            density_node_energy, data["batch"], dim=0, dim_size=n_graphs
        )
        results = dict(base_results)
        results["energy"] = (
            base_results["energy"]
            + electrostatic_energy
            + field_energy
            + density_energy
        )
        status_code = {
            "converged": 0,
            "max_steps_reached": 1,
            "diverged": 2,
            "direct": 3,
        }[state.status]
        results.update(
            {
                "density_coefficients": density,
                "charges": density[:, 0],
                "total_charge": total_charge,
                "dipole": dipole,
                "fermi_level": state.fermi_level,
                "field_features": state.field_features,
                "density_history": state.density_history,
                "electrostatic_energy": electrostatic_energy,
                "density_energy": density_energy,
                "field_energy": field_energy,
                "scf_residual": state.residual,
                "scf_num_steps": density.new_full(
                    (n_graphs,), state.num_steps, dtype=torch.int64
                ),
                "scf_status": density.new_full(
                    (n_graphs,), status_code, dtype=torch.int64
                ),
            }
        )
        return results

    def forward(self, data, graph, base_results):
        descriptors = base_results["descriptors"]
        local_density = self.local_source(descriptors)
        if self.training and self.training_mode == "direct":
            state = self._direct(local_density, descriptors, data, graph)
        else:
            state = self._solve(local_density, descriptors, data, graph)
        if self.training and self.training_mode in {"implicit", "linearize_solve"}:
            state = self._attach_implicit_gradient(
                state, local_density, descriptors, data, graph
            )
        return self._build_results(state, data, graph, base_results)
