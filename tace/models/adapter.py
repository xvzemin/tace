################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Union

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from tace.dataset.element import TorchElement

from ..dataset.quantity import PROPERTY, ComputeFlag
from .lammps import Graph
from .utils import (
    compute_atomic_virials_stresses,
    compute_hessians_vmap,
    compute_symmetric_displacement,
)


class TensorModel(torch.nn.Module):
    def __init__(self, readout_fn: torch.nn.Module):
        super().__init__()
        self.readout_fn = readout_fn
        self.num_fidelities = len(getattr(readout_fn, "fidelity", {})) or 1
        self._set_target_property()
        self._set_lammps_mliap()
        self.reset_fidelity_idx()

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, None]]:
        graph = self.prepare_graph(data)
        RESULTS = self.readout_fn(data, graph)
        FIRST = self.first_derivative_fn(data, graph, RESULTS)
        SECOND = self.second_derivative_fn(data, graph, RESULTS, FIRST)
        OUT = {**RESULTS, **FIRST, **SECOND}
        # return self.denorm(OUT)
        return OUT

    def denorm(
        self, out: Dict[str, Union[torch.Tensor, None]]
    ) -> Dict[str, Union[torch.Tensor, None]]:
        """
        We always return:
        -  normalized value in train
        -  unnormalized value in val/test
        """
        if self.training:
            return out
        for k, v in self.readout_fn.normalizers.items():
            out[k] = v.denorm(out[k])
        if out["e_base_graph"] is not None:
            out["energy"] = out["energy"] + out["e_base_graph"]
        return out

    def first_derivative_fn(
        self, data: Dict[str, Tensor], graph: Graph, results: Dict[str, Tensor]
    ) -> Dict[str, Union[torch.Tensor, None]]:

        E = results["energy"]
        F = None
        V = None
        S = None
        P = None
        M = None
        C_MAG_F = None
        NC_MAG_F = None
        EDGE_F = None
        A_V = None
        A_S = None

        inputs = []
        if self.flags.compute_forces:
            inputs.append(graph.positions)
        if self.flags.compute_stress or self.flags.compute_virials:
            inputs.append(graph.displacement)
        if self.flags.compute_polarization or self.flags.compute_conservative_dipole:
            inputs.append(data["electric_field"])
        # if self.flags.compute_magnetization:
        #     inputs.append(data["magnetic_field"])
        if self.flags.compute_collinear_magnetic_forces:
            inputs.append(data["initial_collinear_magmoms"])
        if self.flags.compute_noncollinear_magnetic_forces:
            inputs.append(data["initial_noncollinear_magmoms"])
        if (
            self.flags.compute_edge_forces
            or self.flags.compute_atomic_virials
            or self.flags.compute_atomic_stresses
        ):
            inputs.append(graph.edge_vector)

        if self.compute_first_derivative:
            grad_outputs = torch.ones_like(E)
            grads = torch.autograd.grad(
                outputs=E,
                inputs=inputs,
                grad_outputs=grad_outputs,
                retain_graph=self.training or self.retain_graph,
                create_graph=self.training or self.create_graph,
                allow_unused=True,
            )
        idx = 0
        if self.flags.compute_forces:
            grad = grads[idx]
            F = torch.zeros_like(data["positions"]) if grad is None else -grad
            idx += 1
        if self.flags.compute_stress or self.flags.compute_virials:
            grad = grads[idx]
            V = torch.zeros_like(data["lattice"]) if grad is None else -grad
            VOLUME = torch.linalg.det(data["lattice"]).abs().unsqueeze(-1)
            S = -V / VOLUME.view(-1, 1, 1)
            S = torch.where(torch.abs(S) < 1e10, S, torch.zeros_like(S))
            idx += 1
        if self.flags.compute_polarization or self.flags.compute_conservative_dipole:
            grad = grads[idx]
            P = torch.zeros_like(data["electric_field"]) if grad is None else -grad
            idx += 1
        # if self.flags.compute_magnetization:
        #     M = -grads[idx]
        #     idx += 1
        if self.flags.compute_collinear_magnetic_forces:
            grad = grads[idx]
            C_MAG_F = (
                torch.zeros_like(data["initial_collinear_magmoms"])
                if grad is None
                else -grad
            )
            idx += 1
        if self.flags.compute_noncollinear_magnetic_forces:
            grad = grads[idx]
            NC_MAG_F = (
                torch.zeros_like(data["initial_noncollinear_magmoms"])
                if grad is None
                else -grad
            )
            idx += 1
        if (
            self.flags.compute_edge_forces
            or self.flags.compute_atomic_virials
            or self.flags.compute_atomic_stresses
        ):
            grad = grads[idx]
            EDGE_F = (
                torch.zeros_like(graph.edge_vector) if grad is None else grad
            )  # consistency with LAMMPS
            idx += 1
            if not self.lmp:
                A_V, A_S = compute_atomic_virials_stresses(
                    graph.edge_vector,
                    EDGE_F,
                    data["edge_index"],
                    graph.lattice,
                    data["batch"],
                    data["node_attrs"].size(0),
                    True,
                    True,
                )

        return {
            "forces": F,
            "virials": V,
            "stress": S,
            "polarization": P,
            "conservative_dipole": P,
            "magnetization": M,
            "collinear_magnetic_forces": C_MAG_F,
            "noncollinear_magnetic_forces": NC_MAG_F,
            "edge_forces": EDGE_F,
            "atomic_virials": A_V,
            "atomic_stresses": A_S,
        }

    def second_derivative_fn(
        self,
        data: Dict[str, torch.Tensor],
        graph: Graph,
        extra: Dict[str, torch.Tensor],
        first_derivative: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        if self.flags.compute_hessian and self.training:
            raise RuntimeError(
                "Hessian prediction is only supported in evaluation mode. "
                "Training with a Hessian target is not supported."
            )

        forces = first_derivative["forces"]
        polarization = first_derivative["polarization"]
        magnetization = first_derivative["magnetization"]

        inputs_list: list[torch.Tensor] = []
        if self.flags.compute_born_effective_charges:
            inputs_list.append(graph.positions)
        if self.flags.compute_conservative_polarizability:
            inputs_list.append(data["electric_field"])

        BEC = None
        ALPHA = None
        CHI_M = None
        HESSIAN = None

        BECList = []
        ALPHAList = []
        # CHI_MList = []
        if (
            self.flags.compute_conservative_polarizability
            or self.flags.compute_born_effective_charges
        ):
            for i in range(3):  # μ = 0,1,2
                polarization_i = polarization.sum(dim=0)[i]  # sum over batch dimension
                grads = torch.autograd.grad(
                    outputs=polarization_i,
                    inputs=inputs_list,
                    retain_graph=(
                        i != 2 or self.training or self.flags.compute_hessian
                    ),
                    create_graph=self.training,
                )
                idx = 0
                if self.flags.compute_born_effective_charges:
                    BEC = grads[idx]
                    idx += 1
                if self.flags.compute_conservative_polarizability:
                    ALPHA = grads[idx]

                if BEC is None:
                    BEC = torch.zeros(
                        graph.positions.shape[0],
                        3,
                        3,
                        device=graph.positions.device,
                        dtype=graph.positions.dtype,
                    )
                if ALPHA is None:
                    ALPHA = torch.zeros_like(data["electric_field"])

                BECList.append(BEC)  # [atoms, 3]
                ALPHAList.append(ALPHA)  # [B,3]

            BEC = torch.stack(BECList, dim=0)  # [3, atoms, 3]
            BEC = BEC.transpose(1, 0)  #        # [atoms, 3 (pol), 3 (pos)]
            ALPHA = torch.stack(ALPHAList, dim=1)  # [B,3,3]

        # if self.flags.compute_magnetization:
        #     MF = data["magnetic_field"]
        #     for i in range(3):
        #         mag_i = magnetization.sum(dim=0)[i]
        #         CHI_M = torch.autograd.grad(
        #             outputs=mag_i,
        #             inputs=[MF],
        #             retain_graph=(
        #                 i != 2
        #                 or self.training
        #                 or self.flags.compute_hessians
        #             ),
        #             create_graph=self.training,
        #             allow_unused=True,
        #         )[0]
        #         if CHI_M is None:
        #             CHI_M = torch.zeros_like(MF)
        #         CHI_MList.append(CHI_M)  # [B,3]
        #     CHI_M = torch.stack(CHI_MList, dim=1)  # [B,3,3]

        # Sampled Hessian training is disabled, only allow eval.
        if self.flags.compute_hessian:
            HESSIAN = compute_hessians_vmap(forces, graph.positions)

        return {
            "hessian": HESSIAN,
            "conservative_polarizability": ALPHA,
            "born_effective_charges": BEC,
            "magnetic_susceptibility": CHI_M,
        }

    def prepare_graph(self, data: Dict[str, torch.Tensor]) -> Graph:

        node_fidelity = (
            data["fidelity_idx"][data["batch"]]
            if ("fidelity_idx" in data)
            else torch.full_like(data["batch"], self.fidelity_idx, dtype=torch.int64)
        )  # used for multi-fidelity and multi-head

        if self.lmp:
            for p in self.get_target_property():
                for requires_grad_p in PROPERTY[p]["requires_grad_with"]:
                    if p != "forces":
                        data[requires_grad_p].requires_grad_(True)
            dtype = data["node_attrs"].dtype
            device = data["node_attrs"].device
            nlocal, nghosts = data["natoms"][0], data["natoms"][1]
            num_graphs = 2
            displacement = None
            positions = torch.zeros(
                (int(nlocal), 3),
                dtype=dtype,
                device=device,
            )
            lattice = torch.zeros(
                (num_graphs, 3, 3),
                dtype=dtype,
                device=device,
            )
            edge_vector = data["edge_vector"].requires_grad_(True)
            edge_length = (edge_vector**2).sum(dim=1, keepdim=True).sqrt() + 1e-9
            lmp_data = data["lmp_data"]
            lmp_natoms = (nlocal, nghosts)
            num_atoms_arange = torch.arange(
                nlocal, device=positions.device, dtype=torch.int64
            )
        else:
            les_module = getattr(self.readout_fn, "les", None)
            if les_module is not None and getattr(les_module, "compute_bec", False):
                data["positions"].requires_grad_(True)
            requires_grad_p_list = []
            for p in self.get_target_property():
                for requires_grad_p in PROPERTY[p]["requires_grad_with"]:
                    requires_grad_p_list.append(requires_grad_p)
                    if requires_grad_p != "edge_vector":
                        data[requires_grad_p].requires_grad_(True)
            dtype = data["node_attrs"].dtype
            device = data["node_attrs"].device
            positions = data["positions"]
            num_graphs = data["ptr"].numel() - 1
            if self.flags.compute_virials or self.flags.compute_stress:
                displacement = compute_symmetric_displacement(data, num_graphs)
            else:
                displacement = None
            source = data["edge_index"][0]
            target = data["edge_index"][1]
            edge_batch = data["batch"][source]
            edge_vector = (
                data["positions"][target]
                - data["positions"][source]
                + torch.einsum(
                    "ni,nij->nj", data["edge_shifts"], data["lattice"][edge_batch]
                )
            )
            if set(self.get_target_property()) & {
                "edge_vector",
                "atomic_stresses",
                "atomic_virials",
            }:
                edge_vector.requires_grad_(True)
            edge_length = (edge_vector**2).sum(dim=1, keepdim=True).sqrt() + 1e-9
            lattice = data["lattice"]
            lmp_data = None
            lmp_natoms = (positions.size(0), 0)
            num_atoms_arange = torch.arange(
                positions.shape[0], device=positions.device, dtype=torch.int64
            )

        # if self.readout_fn.representation.radial_basis.use_dydynamic_cutoff:
        #     dcutoff = self.readout_fn.representation.radial_basis.dydynamic_cutoff_fn(
        #         edge_length,
        #         data["edge_index"],
        #         data['node_attrs'].size(0),
        #     )
        #     mask = edge_length.squeeze(-1) < dcutoff.squeeze(-1)
        #     data['edge_index'] = data['edge_index'][:, mask]
        #     edge_length=edge_length[mask, :]
        #     edge_vector=edge_vector[mask, :]
        #     dcutoff=dcutoff[mask, :]
        # else:
        #     dcutoff = None

        return Graph(
            lmp=self.lmp,
            lmp_data=lmp_data,
            lmp_natoms=lmp_natoms,
            num_graphs=num_graphs,
            displacement=displacement,
            positions=positions,
            edge_vector=edge_vector,
            edge_length=edge_length,
            lattice=lattice,
            node_fidelity=node_fidelity,
            num_atoms_arange=num_atoms_arange,
            # dcutoff=dcutoff,
            dcutoff=None,
            # node_radius=(data["positions"]**2).sum(dim=1, keepdim=True).sqrt() + 1e-9 # TODO
        )

    def get_fidelity_idx(self) -> int:
        return int(self.fidelity_idx)

    def reset_fidelity_idx(self, fidelity_idx: Union[int, None] = 0) -> None:
        if fidelity_idx is not None:
            self.fidelity_idx = fidelity_idx

    def get_embedding_property(self) -> list[str]:
        embedding_property = set(self.readout_fn.embedding_property)
        atomic_basis = getattr(self.readout_fn, "model_config", {}).get(
            "atomic_basis", {}
        )
        atomic_basis_type = atomic_basis.get("type")
        if isinstance(atomic_basis_type, str):
            atomic_basis_type = [atomic_basis_type]
        if atomic_basis_type is not None and any(
            interaction in {"o3_w6j_mag", "o2_mag"} for interaction in atomic_basis_type
        ):
            embedding_property.add("initial_noncollinear_magmoms")
        return list(embedding_property)

    def get_target_property(self) -> list[str]:
        return list(set(self.readout_fn.target_property))

    def _set_target_property(self) -> None:

        target_property = self.get_target_property()

        self.flags = ComputeFlag()
        for k in target_property:
            setattr(self.flags, f"compute_{k}", k in target_property)

        self.compute_first_derivative = False
        for p in target_property:
            if PROPERTY[p]["first_derivative"]:
                self.compute_first_derivative = True

        self.compute_second_derivative = False
        for p in target_property:
            if PROPERTY[p]["second_derivative"]:
                self.compute_second_derivative = True

        self.retain_graph = self.compute_second_derivative
        self.create_graph = self.compute_second_derivative

    def reset_target_property(self, target_property: list[str]) -> None:
        assert isinstance(target_property, list)
        self.readout_fn.target_property = target_property
        self._set_target_property()

    def _set_lammps_mliap(self, enable: bool = False) -> None:
        self.lmp = enable

    def get_model_dtype(self) -> torch.dtype:
        return self.readout_fn.cutoff.dtype

    def get_max_neighbors(self) -> Union[int, None]:
        return self.readout_fn.max_neighbors

    def get_cutoff(self) -> float:
        return float(self.readout_fn.cutoff.item())

    def get_atomic_numbers(self) -> float:
        return [int(z) for z in self.readout_fn.atomic_numbers.cpu().tolist()]

    def get_torch_element(self) -> TorchElement:
        return TorchElement(self.get_atomic_numbers())
