################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Any, Dict, List, Optional

import torch
from e3nn import o3
from e3nn.io import CartesianTensor
from e3nn.o3._reduce import ReducedTensorProducts

from ..linear import e3nnLinear
from .readout import build_scalar_readout, build_tensor_readout


def required_les_irreps(les_arguments: Dict[str, Any]) -> List[str]:
    """Return descriptor irreps required by the enabled LES source terms."""
    required = {"0e"}
    if les_arguments.get("use_dipole", False):
        required.add("1o")
    if les_arguments.get("use_quad", False):
        required.update(("1o", "2e"))
    if les_arguments.get("use_induced_dipole", False) and les_arguments.get(
        "use_anisotropic_polarizability", False
    ):
        alpha_irreps = o3.Irreps(les_arguments.get("alpha_irreps", "0e+1o+2e"))
        required.update(str(ir) for _, ir in alpha_irreps)
    return sorted(required)


class LinearLesReadoutBlock(torch.nn.Module):
    """Predict a symmetric Cartesian tensor from scalar and vector features."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        num_fidelities: int = 1,
        make_w_pos: bool = True,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.num_fidelities = num_fidelities
        self.make_w_pos = make_w_pos
        self.linear = e3nnLinear(self.irreps_in, self.irreps_in)

        self.scalar_slices: List[tuple[int, int]] = []
        self.vector_1o_slices: List[tuple[int, int]] = []
        self.vector_1e_slices: List[tuple[int, int]] = []
        n_scalar = 0
        n_1o = 0
        n_1e = 0
        offset = 0
        for mul, ir in self.irreps_in:
            block_dim = mul * ir.dim
            if ir == o3.Irrep("0e"):
                self.scalar_slices.append((offset, offset + block_dim))
                n_scalar += mul
            elif ir == o3.Irrep("1o"):
                self.vector_1o_slices.append((offset, offset + block_dim))
                n_1o += mul
            elif ir == o3.Irrep("1e"):
                self.vector_1e_slices.append((offset, offset + block_dim))
                n_1e += mul
            offset += block_dim

        if n_scalar == 0:
            raise ValueError("LES tensor readout requires at least one 0e channel")
        if n_1o + n_1e == 0:
            raise ValueError(
                "LES tensor readout requires at least one 1o or 1e channel"
            )

        self.n_1o = n_1o
        self.n_1e = n_1e
        self.scalar_to_weight_1o = (
            torch.nn.Linear(n_scalar, num_fidelities * n_1o) if n_1o > 0 else None
        )
        self.scalar_to_weight_1e = (
            torch.nn.Linear(n_scalar, num_fidelities * n_1e) if n_1e > 0 else None
        )

    def _collect_scalars(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, start:end] for start, end in self.scalar_slices], dim=-1)

    @staticmethod
    def _collect_vectors(
        x: torch.Tensor, slices: List[tuple[int, int]]
    ) -> torch.Tensor:
        return torch.cat(
            [x[:, start:end].reshape(x.shape[0], -1, 3) for start, end in slices],
            dim=1,
        )

    @staticmethod
    def _dyadic_sum(weights: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nfc,nci,ncj->nfij", weights, vectors, vectors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        scalars = self._collect_scalars(y)
        tensor = y.new_zeros((y.shape[0], self.num_fidelities, 3, 3))

        if self.scalar_to_weight_1o is not None:
            vectors = self._collect_vectors(y, self.vector_1o_slices)
            weights = self.scalar_to_weight_1o(scalars).reshape(
                y.shape[0], self.num_fidelities, self.n_1o
            )
            if self.make_w_pos:
                weights = weights.square()
            tensor = tensor + self._dyadic_sum(weights, vectors)

        if self.scalar_to_weight_1e is not None:
            vectors = self._collect_vectors(y, self.vector_1e_slices)
            weights = self.scalar_to_weight_1e(scalars).reshape(
                y.shape[0], self.num_fidelities, self.n_1e
            )
            tensor = tensor + self._dyadic_sum(weights, vectors)

        return tensor


class NonLinearLesReadoutBlock(torch.nn.Module):
    """Predict a symmetric tensor using scalar-conditioned vector mixing."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        num_fidelities: int = 1,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.num_fidelities = num_fidelities
        self.linear = e3nnLinear(self.irreps_in, self.irreps_in)

        self.scalar_slices: List[tuple[int, int]] = []
        self.vector_1o_slices: List[tuple[int, int]] = []
        n_scalar = 0
        n_1o = 0
        offset = 0
        for mul, ir in self.irreps_in:
            block_dim = mul * ir.dim
            if ir == o3.Irrep("0e"):
                self.scalar_slices.append((offset, offset + block_dim))
                n_scalar += mul
            elif ir == o3.Irrep("1o"):
                self.vector_1o_slices.append((offset, offset + block_dim))
                n_1o += mul
            offset += block_dim

        if n_scalar == 0:
            raise ValueError("nonlinear LES tensor readout requires 0e channels")
        if n_1o == 0:
            raise ValueError("nonlinear LES tensor readout requires 1o channels")

        self.n_1o = n_1o
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_scalar, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_fidelities * n_1o * n_1o),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        scalars = torch.cat(
            [y[:, start:end] for start, end in self.scalar_slices], dim=-1
        )
        vectors = torch.cat(
            [
                y[:, start:end].reshape(y.shape[0], -1, 3)
                for start, end in self.vector_1o_slices
            ],
            dim=1,
        )
        mixing = self.mlp(scalars).reshape(
            y.shape[0], self.num_fidelities, self.n_1o, self.n_1o
        )
        mixing = 0.5 * (mixing + mixing.transpose(-1, -2))
        tensor = torch.einsum("nfcd,nci,ndj->nfij", mixing, vectors, vectors)
        return 0.5 * (tensor + tensor.transpose(-1, -2))


class _L2ToCartesian(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        basis = ReducedTensorProducts(
            "ij=ji", i="1o", filter_ir_out=["2e"]
        ).change_of_basis
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.einsum("kij,...k->...ij", self.basis, tensor)


class _SymmetricToCartesian(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        basis = CartesianTensor("ij=ji").reduced_tensor_products().change_of_basis
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.einsum("kij,...k->...ij", self.basis, tensor)


class TACELes(torch.nn.Module):
    """Complete TACE readout and adapter for the external LES solver."""

    def __init__(
        self,
        backend: torch.nn.Module,
        les_arguments: Dict[str, Any],
        *,
        num_layers: int,
        hidden_channel: List[int],
        bias: bool,
        num_elements: int,
        num_fidelities: int,
        use_alllayer: bool,
        parity: bool,
        irreps_in: List[o3.Irreps],
    ) -> None:
        super().__init__()
        self.backend = backend
        self.num_fidelities = num_fidelities
        self.descriptor_indices = list(range(num_layers)) if use_alllayer else [-1]

        self.compute_bec = bool(les_arguments.get("compute_bec", False))
        self.bec_output_index = les_arguments.get("bec_output_index")
        self.use_dipoles = bool(les_arguments.get("use_dipole", False))
        self.use_quads = bool(les_arguments.get("use_quad", False))
        self.use_induced_charges = bool(les_arguments.get("use_induced_charge", False))
        self.use_induced_dipoles = bool(les_arguments.get("use_induced_dipole", False))
        self.use_anisotropic_polarizability = bool(
            les_arguments.get("use_anisotropic_polarizability", False)
        )
        self.alpha_irreps = o3.Irreps(les_arguments.get("alpha_irreps", "0e+1o+2e"))
        self.make_alpha_positive = bool(les_arguments.get("make_alpha_positive", False))
        self.make_kappa_positive = bool(les_arguments.get("make_kappa_positive", False))
        self.output_scale = float(les_arguments.get("output_scale", 0.1))
        self.kappa_scale = float(les_arguments.get("kappa_scale", 0.01))
        self.alpha_scale = float(les_arguments.get("alpha_scale", 0.01))

        scalar_kwargs = {
            "num_layers": num_layers,
            "hidden_channel": hidden_channel,
            "bias": bias,
            "num_elements": num_elements,
            "num_fidelities": num_fidelities,
            "use_alllayer": use_alllayer,
            "parity": parity,
            "irreps_in": irreps_in,
        }
        tensor_kwargs = dict(scalar_kwargs)
        self.charge_readouts = build_scalar_readout(irreps_out="0e", **scalar_kwargs)

        self.dipole_readouts = (
            build_tensor_readout(irreps_out="1o", **tensor_kwargs)
            if self.use_dipoles
            else torch.nn.ModuleList()
        )
        self.kappa_readouts = (
            build_scalar_readout(irreps_out="0e", **scalar_kwargs)
            if self.use_induced_charges
            else torch.nn.ModuleList()
        )

        self.quad_2e_readouts = torch.nn.ModuleList()
        self.quad_dyadic_readouts = torch.nn.ModuleList()
        if self.use_quads:
            self.quad_2e_readouts = build_tensor_readout(
                irreps_out="2e", **tensor_kwargs
            )
            self.quad_dyadic_readouts = self._build_dyadic_readouts(
                irreps_in,
                use_alllayer,
                num_fidelities,
                nonlinear=False,
                make_w_pos=False,
            )

        self.alpha_scalar_readouts = torch.nn.ModuleList()
        self.alpha_2e_scalar_readouts = torch.nn.ModuleList()
        self.alpha_2e_readouts = torch.nn.ModuleList()
        self.alpha_dyadic_readouts = torch.nn.ModuleList()
        if self.use_induced_dipoles:
            if not self.use_anisotropic_polarizability or "0e" in self.alpha_irreps:
                self.alpha_scalar_readouts = build_scalar_readout(
                    irreps_out="0e", **scalar_kwargs
                )
            if self.use_anisotropic_polarizability and "2e" in self.alpha_irreps:
                self.alpha_2e_scalar_readouts = build_scalar_readout(
                    irreps_out="0e", **scalar_kwargs
                )
                self.alpha_2e_readouts = build_tensor_readout(
                    irreps_out="2e", **tensor_kwargs
                )
            if self.use_anisotropic_polarizability and (
                "1o" in self.alpha_irreps or "1e" in self.alpha_irreps
            ):
                self.alpha_dyadic_readouts = self._build_dyadic_readouts(
                    irreps_in,
                    use_alllayer,
                    num_fidelities,
                    nonlinear=bool(
                        les_arguments.get("alpha_1o_nonlinear_readout", False)
                    ),
                    make_w_pos=bool(les_arguments.get("alpha_1o_linear_w_pos", True)),
                )

        self.l2_to_cartesian = _L2ToCartesian()
        self.symmetric_to_cartesian = _SymmetricToCartesian()

    @staticmethod
    def _build_dyadic_readouts(
        irreps_in: List[o3.Irreps],
        use_alllayer: bool,
        num_fidelities: int,
        *,
        nonlinear: bool,
        make_w_pos: bool,
    ) -> torch.nn.ModuleList:
        selected_irreps = irreps_in if use_alllayer else [irreps_in[-1]]
        block = NonLinearLesReadoutBlock if nonlinear else LinearLesReadoutBlock
        readouts = torch.nn.ModuleList()
        for irreps in selected_irreps:
            if nonlinear:
                readouts.append(block(irreps, num_fidelities=num_fidelities))
            else:
                readouts.append(
                    block(
                        irreps,
                        num_fidelities=num_fidelities,
                        make_w_pos=make_w_pos,
                    )
                )
        return readouts

    @staticmethod
    def _select_fidelity(
        tensor: torch.Tensor,
        atom_indices: torch.Tensor,
        node_fidelity: torch.Tensor,
    ) -> torch.Tensor:
        local_fidelity = node_fidelity.index_select(0, atom_indices)
        local_tensor = tensor.index_select(0, atom_indices)
        local_indices = torch.arange(
            atom_indices.numel(), device=atom_indices.device, dtype=torch.int64
        )
        return local_tensor[local_indices, local_fidelity]

    def _sum_scalar(
        self,
        readouts: torch.nn.ModuleList,
        descriptors: List[torch.Tensor],
        atom_indices: torch.Tensor,
        node_fidelity: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if len(readouts) == 0:
            return None
        values = []
        for readout, descriptor_index in zip(readouts, self.descriptor_indices):
            out = readout(descriptors[descriptor_index], node_fidelity)
            values.append(self._select_fidelity(out, atom_indices, node_fidelity))
        return torch.stack(values, dim=0).sum(dim=0)

    def _sum_irrep(
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
        for readout, descriptor_index in zip(readouts, self.descriptor_indices):
            out = readout(descriptors[descriptor_index], node_fidelity).reshape(
                descriptors[descriptor_index].shape[0], self.num_fidelities, irrep_dim
            )
            values.append(self._select_fidelity(out, atom_indices, node_fidelity))
        return torch.stack(values, dim=0).sum(dim=0)

    def _sum_dyadic(
        self,
        readouts: torch.nn.ModuleList,
        descriptors: List[torch.Tensor],
        atom_indices: torch.Tensor,
        node_fidelity: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if len(readouts) == 0:
            return None
        values = []
        for readout, descriptor_index in zip(readouts, self.descriptor_indices):
            out = readout(descriptors[descriptor_index])
            values.append(self._select_fidelity(out, atom_indices, node_fidelity))
        return torch.stack(values, dim=0).sum(dim=0)

    @staticmethod
    def _sum_optional(values: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        present = [value for value in values if value is not None]
        if not present:
            return None
        return torch.stack(present, dim=0).sum(dim=0)

    def _run_backend(
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
        common = {
            "compute_energy": True,
            "compute_bec": self.compute_bec,
            "bec_output_index": self.bec_output_index,
        }
        if external_field is None or external_field.ndim == 1:
            return self.backend(
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
            return self.backend(
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
        graph_results: List[tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]] = []
        split_common = dict(common)
        split_common["compute_bec"] = False
        for graph_index in range(cell.shape[0]):
            mask = batch == graph_index
            local_batch = torch.zeros(
                int(mask.sum()), device=batch.device, dtype=torch.int64
            )
            result = self.backend(
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
        energy_parts = [result["E_lr"] for _, result in graph_results]
        output["E_lr"] = torch.cat(
            [energy for energy in energy_parts if energy is not None], dim=0
        )
        for key in (
            "latent_charges",
            "latent_dipoles",
            "latent_quads",
            "latent_alphas",
            "BEC",
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
        if self.compute_bec:
            output["BEC"] = self.backend.bec(
                q=output["latent_charges"],
                u=output["latent_dipoles"],
                r=positions,
                cell=cell,
                batch=batch,
                output_index=self.bec_output_index,
            )
        return output

    def forward(
        self,
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
        latent_charges = self._sum_scalar(
            self.charge_readouts, descriptors, atom_indices, node_fidelity
        )
        if latent_charges is None:
            raise RuntimeError("LES charge readout was not constructed")
        latent_charges = latent_charges * self.output_scale

        latent_dipoles = self._sum_irrep(
            self.dipole_readouts,
            descriptors,
            atom_indices,
            node_fidelity,
            3,
        )
        if latent_dipoles is not None:
            latent_dipoles = latent_dipoles * self.output_scale

        quad_2e = self._sum_irrep(
            self.quad_2e_readouts,
            descriptors,
            atom_indices,
            node_fidelity,
            5,
        )
        quad_2e_cartesian = (
            self.l2_to_cartesian(quad_2e) if quad_2e is not None else None
        )
        quad_dyadic = self._sum_dyadic(
            self.quad_dyadic_readouts,
            descriptors,
            atom_indices,
            node_fidelity,
        )
        latent_quads = self._sum_optional([quad_2e_cartesian, quad_dyadic])
        if latent_quads is not None:
            trace = latent_quads.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
            eye = torch.eye(3, device=latent_quads.device, dtype=latent_quads.dtype)
            latent_quads = latent_quads - trace[:, None, None] * eye[None] / 3.0
            latent_quads = latent_quads * self.output_scale

        latent_kappas = self._sum_scalar(
            self.kappa_readouts, descriptors, atom_indices, node_fidelity
        )
        if latent_kappas is not None:
            latent_kappas = latent_kappas * self.kappa_scale
            if self.make_kappa_positive:
                latent_kappas = latent_kappas.square()

        alpha_scalar = self._sum_scalar(
            self.alpha_scalar_readouts, descriptors, atom_indices, node_fidelity
        )
        alpha_2e_scalar = self._sum_scalar(
            self.alpha_2e_scalar_readouts,
            descriptors,
            atom_indices,
            node_fidelity,
        )
        alpha_2e = self._sum_irrep(
            self.alpha_2e_readouts,
            descriptors,
            atom_indices,
            node_fidelity,
            5,
        )
        alpha_2e_cartesian = None
        if alpha_2e_scalar is not None and alpha_2e is not None:
            alpha_2e_cartesian = self.symmetric_to_cartesian(
                torch.cat((alpha_2e_scalar[:, None], alpha_2e), dim=-1)
            )
        alpha_dyadic = self._sum_dyadic(
            self.alpha_dyadic_readouts,
            descriptors,
            atom_indices,
            node_fidelity,
        )

        if self.use_anisotropic_polarizability:
            alpha_scalar_cartesian = None
            if alpha_scalar is not None:
                eye = torch.eye(3, device=alpha_scalar.device, dtype=alpha_scalar.dtype)
                alpha_scalar_cartesian = alpha_scalar[:, None, None] * eye[None]
            latent_alphas = self._sum_optional(
                [alpha_scalar_cartesian, alpha_2e_cartesian, alpha_dyadic]
            )
        else:
            latent_alphas = alpha_scalar

        if latent_alphas is not None:
            latent_alphas = latent_alphas * self.alpha_scale
            if self.make_alpha_positive:
                if latent_alphas.ndim == 1:
                    latent_alphas = latent_alphas.square()
                else:
                    latent_alphas = torch.matmul(
                        latent_alphas, latent_alphas.transpose(-1, -2)
                    )

        local_positions = positions.index_select(0, atom_indices)
        local_batch = batch.index_select(0, atom_indices)
        local_atomic_numbers = atomic_numbers.index_select(0, atom_indices)
        if self.compute_bec and latent_dipoles is not None:
            latent_dipoles = latent_dipoles + 0.0 * local_positions.sum()
        cell_les = cell.clone()
        if pbc is not None:
            nonperiodic = ~pbc.to(torch.bool).any(dim=-1)
            cell_les[nonperiodic] = 0.0

        result = self._run_backend(
            atomic_numbers=local_atomic_numbers,
            latent_charges=latent_charges,
            latent_dipoles=latent_dipoles,
            latent_quads=latent_quads,
            latent_alphas=latent_alphas,
            latent_kappas=latent_kappas,
            positions=local_positions,
            cell=cell_les,
            batch=local_batch,
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
