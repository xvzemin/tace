################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Any, Dict, List, Optional

import torch

from .basis_change import LESPolarizability, LESQuadrupoles
from .readout import build_scalar_readout, build_tensor_readout


def required_les_irreps(les_arguments: Dict[str, Any]) -> List[str]:
    """Return descriptor irreps required by the enabled LES source terms."""
    required = {"0e"}
    if les_arguments.get("use_dipole", False):
        required.add("1o")
    if les_arguments.get("use_quad", False) or (
        les_arguments.get("use_induced_dipole", False)
        and les_arguments.get("use_anisotropic_polarizability", False)
    ):
        required.add("2e")
    return sorted(required)


class TACELES(torch.nn.Module):
    """Read out TACE latent sources and pass them to the LES solver."""

    def __init__(
        self,
        les_arguments: Dict[str, Any],
        *,
        num_layers: int,
        hidden_channel: List[int],
        bias: bool,
        num_elements: int,
        num_fidelities: int,
        use_alllayer: bool,
        parity: bool,
        irreps_in: List[Any],
    ) -> None:
        super().__init__()
        try:
            from les import Les
        except ImportError as error:
            raise ImportError(
                "LES is required but could not be imported. "
                "Please install the version of LES specified by TACE. "
                "TACE guide: https://tace.readthedocs.io/en/latest/guide/les.html "
                "LES repository: https://github.com/ChengUCB/les"
            ) from error
        self.les = Les(les_arguments=les_arguments)

        readout_kwargs = {
            "num_layers": num_layers,
            "hidden_channel": hidden_channel,
            "bias": bias,
            "num_elements": num_elements,
            "num_fidelities": num_fidelities,
            "use_alllayer": use_alllayer,
            "parity": parity,
            "irreps_in": irreps_in,
        }
        self.num_layers = num_layers
        self.num_fidelities = num_fidelities
        self.use_alllayer = use_alllayer
        self.charge_readouts = build_scalar_readout(
            irreps_out="0e", **readout_kwargs
        )
        self.dipole_readouts = torch.nn.ModuleList()
        self.quad_readouts = torch.nn.ModuleList()
        self.kappa_readouts = torch.nn.ModuleList()
        self.alpha_scalar_readouts = torch.nn.ModuleList()
        self.alpha_tensor_readouts = torch.nn.ModuleList()

        if les_arguments.get("use_dipole", False):
            self.dipole_readouts = build_tensor_readout(
                irreps_out="1o", **readout_kwargs
            )
        if les_arguments.get("use_quad", False):
            self.quad_readouts = build_tensor_readout(
                irreps_out="2e", **readout_kwargs
            )
            self.quad_basis_change = LESQuadrupoles()
        if les_arguments.get("use_induced_charge", False):
            self.kappa_readouts = build_scalar_readout(
                irreps_out="0e", **readout_kwargs
            )
        if les_arguments.get("use_induced_dipole", False):
            self.alpha_scalar_readouts = build_scalar_readout(
                irreps_out="0e", **readout_kwargs
            )
            if les_arguments.get("use_anisotropic_polarizability", False):
                self.alpha_tensor_readouts = build_tensor_readout(
                    irreps_out="2e", **readout_kwargs
                )
                self.alpha_basis_change = LESPolarizability()

        self.make_alpha_positive = bool(
            les_arguments.get("make_alpha_positive", False)
        )
        self.make_kappa_positive = bool(
            les_arguments.get("make_kappa_positive", False)
        )
        self.kappa_scale = float(les_arguments.get("kappa_scale", 0.01))
        self.alpha_scale = float(les_arguments.get("alpha_scale", 0.01))

    def _sum_readouts(
        self,
        readouts: torch.nn.ModuleList,
        descriptors: List[torch.Tensor],
        atom_indices: torch.Tensor,
        node_fidelity: torch.Tensor,
        irrep_dim: int,
    ) -> Optional[torch.Tensor]:
        if len(readouts) == 0:
            return None
        values = []
        descriptor_indices = range(self.num_layers) if self.use_alllayer else [-1]
        for readout, descriptor_index in zip(readouts, descriptor_indices):
            output = readout(
                descriptors[descriptor_index], node_fidelity
            ).reshape(
                descriptors[descriptor_index].shape[0],
                self.num_fidelities,
                irrep_dim,
            )
            local_output = output.index_select(0, atom_indices)
            local_fidelity = node_fidelity.index_select(0, atom_indices)
            local_indices = torch.arange(
                atom_indices.numel(), device=atom_indices.device, dtype=torch.int64
            )
            values.append(local_output[local_indices, local_fidelity])
        output = torch.stack(values, dim=0).sum(dim=0)
        return output.squeeze(-1) if irrep_dim == 1 else output

    def _readout_sources(
        self,
        descriptors: List[torch.Tensor],
        node_fidelity: torch.Tensor,
        atom_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        latent_charges = self._sum_readouts(
            self.charge_readouts, descriptors, atom_indices, node_fidelity, 1
        )
        latent_charges = latent_charges * self.les.output_scaling_factor

        latent_dipoles = self._sum_readouts(
            self.dipole_readouts, descriptors, atom_indices, node_fidelity, 3
        )
        if latent_dipoles is not None:
            latent_dipoles = latent_dipoles * self.les.output_scaling_factor

        latent_quads = self._sum_readouts(
            self.quad_readouts, descriptors, atom_indices, node_fidelity, 5
        )
        if latent_quads is not None:
            latent_quads = self.quad_basis_change(
                latent_quads.new_zeros(latent_quads.shape[0]), latent_quads
            )
            latent_quads = latent_quads * self.les.output_scaling_factor

        latent_kappas = self._sum_readouts(
            self.kappa_readouts, descriptors, atom_indices, node_fidelity, 1
        )
        if latent_kappas is not None:
            latent_kappas = latent_kappas * self.kappa_scale
            if self.make_kappa_positive:
                latent_kappas = latent_kappas.square()

        latent_alpha_scalar = self._sum_readouts(
            self.alpha_scalar_readouts, descriptors, atom_indices, node_fidelity, 1
        )
        latent_alpha_tensor = self._sum_readouts(
            self.alpha_tensor_readouts, descriptors, atom_indices, node_fidelity, 5
        )
        if latent_alpha_tensor is not None:
            latent_alphas = self.alpha_basis_change(
                latent_alpha_scalar, latent_alpha_tensor
            )
        else:
            latent_alphas = latent_alpha_scalar
        if latent_alphas is not None:
            latent_alphas = latent_alphas * self.alpha_scale
            if self.make_alpha_positive:
                if latent_alphas.ndim == 1:
                    latent_alphas = latent_alphas.square()
                else:
                    latent_alphas = torch.matmul(
                        latent_alphas, latent_alphas.transpose(-1, -2)
                    )

        return (
            latent_charges,
            latent_dipoles,
            latent_quads,
            latent_alphas,
            latent_kappas,
        )

    def _les_options(self) -> Dict[str, Any]:
        return {
            "compute_energy": True,
            "compute_bec": bool(getattr(self.les, "compute_bec", False)),
            "bec_output_index": getattr(self.les, "bec_output_index", None),
        }

    def _solve(self, **kwargs) -> Dict[str, Optional[torch.Tensor]]:
        return self.les(**kwargs)

    def _run_les(
        self,
        *,
        atomic_numbers: torch.Tensor,
        latent_charges: torch.Tensor,
        latent_dipoles: Optional[torch.Tensor],
        latent_quads: Optional[torch.Tensor],
        latent_alphas: Optional[torch.Tensor],
        latent_kappas: Optional[torch.Tensor],
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch: torch.Tensor,
        external_field: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        common = self._les_options()
        if external_field is None or external_field.ndim == 1:
            return self._solve(
                atomic_numbers=atomic_numbers,
                latent_charges=latent_charges,
                latent_dipoles=latent_dipoles,
                latent_quads=latent_quads,
                latent_alphas=latent_alphas,
                latent_kappas=latent_kappas,
                positions=positions,
                cell=cell,
                batch=batch,
                e_ext=external_field,
                **common,
            )

        if external_field.ndim != 2 or external_field.shape != (cell.shape[0], 3):
            raise ValueError(
                "electric_field must have shape [3] or [num_graphs, 3] for LES"
            )
        if torch.equal(external_field, external_field[:1].expand_as(external_field)):
            return self._solve(
                atomic_numbers=atomic_numbers,
                latent_charges=latent_charges,
                latent_dipoles=latent_dipoles,
                latent_quads=latent_quads,
                latent_alphas=latent_alphas,
                latent_kappas=latent_kappas,
                positions=positions,
                cell=cell,
                batch=batch,
                e_ext=external_field[0],
                **common,
            )

        split_common = dict(common)
        split_common["compute_bec"] = False
        graph_results: List[tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]] = []
        for graph_index in range(cell.shape[0]):
            mask = batch == graph_index
            local_batch = torch.zeros(
                int(mask.sum()), device=batch.device, dtype=torch.int64
            )
            result = self._solve(
                atomic_numbers=atomic_numbers[mask],
                latent_charges=latent_charges[mask],
                latent_dipoles=(
                    latent_dipoles[mask] if latent_dipoles is not None else None
                ),
                latent_quads=latent_quads[mask] if latent_quads is not None else None,
                latent_alphas=(
                    latent_alphas[mask] if latent_alphas is not None else None
                ),
                latent_kappas=(
                    latent_kappas[mask] if latent_kappas is not None else None
                ),
                positions=positions[mask],
                cell=cell[graph_index : graph_index + 1],
                batch=local_batch,
                e_ext=external_field[graph_index],
                **split_common,
            )
            graph_results.append((mask, result))

        output: Dict[str, Optional[torch.Tensor]] = {}
        output["E_lr"] = torch.cat(
            [result["E_lr"] for _, result in graph_results], dim=0
        )
        for key in (
            "latent_charges",
            "latent_dipoles",
            "latent_quads",
            "latent_alphas",
        ):
            first = next(
                (result[key] for _, result in graph_results if result[key] is not None),
                None,
            )
            if first is None:
                output[key] = None
                continue
            combined = first.new_empty((positions.shape[0], *first.shape[1:]))
            for mask, result in graph_results:
                value = result[key]
                if value is not None:
                    combined[mask] = value
            output[key] = combined

        output["BEC"] = None
        if common["compute_bec"]:
            output["BEC"] = self.les.bec(
                q=output["latent_charges"],
                u=output["latent_dipoles"],
                r=positions,
                cell=cell,
                batch=batch,
                output_index=common["bec_output_index"],
            )
        return output

    def forward(
        self,
        *,
        descriptors: List[torch.Tensor],
        node_fidelity: torch.Tensor,
        atom_indices: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch: torch.Tensor,
        atomic_numbers: torch.Tensor,
        external_field: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        (
            latent_charges,
            latent_dipoles,
            latent_quads,
            latent_alphas,
            latent_kappas,
        ) = self._readout_sources(descriptors, node_fidelity, atom_indices)
        positions = positions.index_select(0, atom_indices)
        batch = batch.index_select(0, atom_indices)
        atomic_numbers = atomic_numbers.index_select(0, atom_indices)
        if (
            bool(getattr(self.les, "compute_bec", False))
            and latent_dipoles is not None
        ):
            latent_dipoles = latent_dipoles + 0.0 * positions.sum()

        cell_les = cell.clone()
        if pbc is not None:
            nonperiodic = ~pbc.to(torch.bool).any(dim=-1)
            cell_les[nonperiodic] = 0.0

        result = self._run_les(
            atomic_numbers=atomic_numbers,
            latent_charges=latent_charges,
            latent_dipoles=latent_dipoles,
            latent_quads=latent_quads,
            latent_alphas=latent_alphas,
            latent_kappas=latent_kappas,
            positions=positions,
            cell=cell_les,
            batch=batch,
            external_field=external_field,
        )
        return {
            "les_energy": result["E_lr"],
            "les_latent_charges": result["latent_charges"],
            "les_latent_dipoles": result["latent_dipoles"],
            "les_latent_quadrupoles": result["latent_quads"],
            "les_latent_polarizabilities": result["latent_alphas"],
            "les_latent_kappas": latent_kappas,
            "les_born_effective_charges": result["BEC"],
        }
