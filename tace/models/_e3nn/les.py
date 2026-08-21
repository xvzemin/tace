################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Any, Dict, List, Optional

import torch

from .basis_change import LESLatentAlphas, LESLatentQuads
from .readout import build_scalar_readout, build_tensor_readout


def required_les_irreps(les_arguments: Dict[str, Any]) -> List[str]:
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
    """Predict latent quantities with TACE readouts and pass them to LES."""

    batch_external_field = True
    scale = {
        "latent_charges": 0.1,
        "latent_dipoles": 0.1,
        "latent_quads": 0.1,
        "latent_kappas": 0.01,
        "latent_alphas": 0.01,
    }

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
        self.latent_charges_readouts = build_scalar_readout(
            irreps_out="0e", **readout_kwargs
        )

        if les_arguments.get("use_dipole", False):
            self.latent_dipoles_readouts = build_tensor_readout(
                irreps_out="1o", **readout_kwargs
            )
        if les_arguments.get("use_quad", False):
            self.latent_quads_readouts = build_tensor_readout(
                irreps_out="2e", **readout_kwargs
            )
            self.latent_quads_basis_change = LESLatentQuads()
        if les_arguments.get("use_induced_charge", False):
            self.latent_kappas_readouts = build_scalar_readout(
                irreps_out="0e", **readout_kwargs
            )
        if les_arguments.get("use_induced_dipole", False):
            self.latent_alphas_readout0s = build_scalar_readout(
                irreps_out="0e", **readout_kwargs
            )
            if les_arguments.get("use_anisotropic_polarizability", False):
                self.latent_alphas_readout2s = build_tensor_readout(
                    irreps_out="2e", **readout_kwargs
                )
                self.latent_alphas_basis_change = LESLatentAlphas()

        self.make_latent_alphas_positive = bool(
            les_arguments.get("make_alpha_positive", False)
        )
        self.make_latent_kappas_positive = bool(
            les_arguments.get("make_kappa_positive", False)
        )
    def _sum_readouts(
        self,
        readouts: torch.nn.ModuleList,
        descriptors: List[torch.Tensor],
        num_atoms_arange: torch.Tensor,
        node_fidelity: torch.Tensor,
        irrep_dim: int,
    ) -> torch.Tensor:
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
            local_output = output.index_select(0, num_atoms_arange)
            local_fidelity = node_fidelity.index_select(0, num_atoms_arange)
            local_indices = torch.arange(
                num_atoms_arange.numel(),
                device=num_atoms_arange.device,
                dtype=torch.int64,
            )
            values.append(local_output[local_indices, local_fidelity])
        output = torch.stack(values, dim=0).sum(dim=0)
        return output.squeeze(-1) if irrep_dim == 1 else output

    def _readout_latent_quantities(
        self,
        descriptors: List[torch.Tensor],
        node_fidelity: torch.Tensor,
        num_atoms_arange: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        latent_charges = self._sum_readouts(
            self.latent_charges_readouts,
            descriptors,
            num_atoms_arange,
            node_fidelity,
            1,
        )
        latent_charges = latent_charges * self.scale["latent_charges"]

        latent_dipoles = None
        if hasattr(self, "latent_dipoles_readouts"):
            latent_dipoles = self._sum_readouts(
                self.latent_dipoles_readouts,
                descriptors,
                num_atoms_arange,
                node_fidelity,
                3,
            )
            latent_dipoles = latent_dipoles * self.scale["latent_dipoles"]

        latent_quads = None
        if hasattr(self, "latent_quads_readouts"):
            latent_quads = self._sum_readouts(
                self.latent_quads_readouts,
                descriptors,
                num_atoms_arange,
                node_fidelity,
                5,
            )
            latent_quads = self.latent_quads_basis_change(
                latent_quads.new_zeros(latent_quads.shape[0]), latent_quads
            )
            latent_quads = latent_quads * self.scale["latent_quads"]

        latent_kappas = None
        if hasattr(self, "latent_kappas_readouts"):
            latent_kappas = self._sum_readouts(
                self.latent_kappas_readouts,
                descriptors,
                num_atoms_arange,
                node_fidelity,
                1,
            )
            latent_kappas = latent_kappas * self.scale["latent_kappas"]
            if self.make_latent_kappas_positive:
                latent_kappas = latent_kappas.square()

        latent_alphas = None
        if hasattr(self, "latent_alphas_readout0s"):
            latent_alphas = self._sum_readouts(
                self.latent_alphas_readout0s,
                descriptors,
                num_atoms_arange,
                node_fidelity,
                1,
            )
            if hasattr(self, "latent_alphas_readout2s"):
                latent_alpha2 = self._sum_readouts(
                    self.latent_alphas_readout2s,
                    descriptors,
                    num_atoms_arange,
                    node_fidelity,
                    5,
                )
                latent_alphas = self.latent_alphas_basis_change(
                    latent_alphas, latent_alpha2
                )
            latent_alphas = latent_alphas * self.scale["latent_alphas"]
            if self.make_latent_alphas_positive:
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
        lattice: torch.Tensor,
        batch: torch.Tensor,
        external_field: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        common = self._les_options()
        if external_field is None or not self.batch_external_field:
            return self._solve(
                atomic_numbers=atomic_numbers,
                latent_charges=latent_charges,
                latent_dipoles=latent_dipoles,
                latent_quads=latent_quads,
                latent_alphas=latent_alphas,
                latent_kappas=latent_kappas,
                positions=positions,
                cell=lattice,
                batch=batch,
                e_ext=None if external_field is None else external_field[0],
                **common,
            )

        split_common = dict(common)
        split_common["compute_bec"] = False
        graph_results: List[tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]] = []
        for graph_index in range(lattice.shape[0]):
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
                cell=lattice[graph_index : graph_index + 1],
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
                cell=lattice,
                batch=batch,
                output_index=common["bec_output_index"],
            )
        return output

    def forward(
        self,
        *,
        descriptors: List[torch.Tensor],
        node_fidelity: torch.Tensor,
        num_atoms_arange: torch.Tensor,
        positions: torch.Tensor,
        lattice: torch.Tensor,
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
        ) = self._readout_latent_quantities(
            descriptors, node_fidelity, num_atoms_arange
        )
        positions = positions.index_select(0, num_atoms_arange)
        batch = batch.index_select(0, num_atoms_arange)
        atomic_numbers = atomic_numbers.index_select(0, num_atoms_arange)
        if (
            bool(getattr(self.les, "compute_bec", False))
            and latent_dipoles is not None
        ):
            latent_dipoles = latent_dipoles + 0.0 * positions.sum()

        lattice_les = lattice.clone()
        if pbc is not None:
            nonperiodic = ~pbc.to(torch.bool).any(dim=-1)
            lattice_les[nonperiodic] = 0.0

        result = self._run_les(
            atomic_numbers=atomic_numbers,
            latent_charges=latent_charges,
            latent_dipoles=latent_dipoles,
            latent_quads=latent_quads,
            latent_alphas=latent_alphas,
            latent_kappas=latent_kappas,
            positions=positions,
            lattice=lattice_les,
            batch=batch,
            external_field=external_field,
        )
        return {
            "les_energy": result["E_lr"],
            "les_latent_charges": result["latent_charges"],
            "les_latent_dipoles": result["latent_dipoles"],
            "les_latent_quads": result["latent_quads"],
            "les_latent_alphas": result["latent_alphas"],
            "les_latent_kappas": latent_kappas,
            "les_born_effective_charges": result["BEC"],
        }
