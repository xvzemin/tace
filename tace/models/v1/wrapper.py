################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Dict, Optional, Tuple


import torch
from torch import Tensor
from e3nn.util.jit import compile_mode

from .utils import Graph
from ...dataset.quantity import PROPERTY, ComputeFlag
from ...utils.torch_scatter import scatter_sum


@compile_mode("script")
class WrapModelV1(torch.nn.Module):
    def __init__(self, readout_fn: torch.nn.Module):
        super().__init__()
        # === init and compute flag ===
        target = set(readout_fn.target_property)
        self.target_property = readout_fn.target_property
        self.embedding_property = readout_fn.embedding_property
        self.conservations = readout_fn.conservations
        self.universal_embedding = readout_fn.universal_embedding
        self.enable_electric_field = readout_fn.enable_electric_field
        self.enable_magnetic_field = readout_fn.enable_magnetic_field
        self.readout_fn = readout_fn
        self.lmp = False
        self.flags = ComputeFlag()
        for k in self.target_property:
            setattr(self.flags, f"compute_{k}", k in self.target_property)

        self.compute_first_derivative = False
        for p in self.target_property:
            if PROPERTY[p]['first_derivative']:
                self.compute_first_derivative = True

        self.compute_second_derivative = False
        for p in self.target_property:
            if PROPERTY[p]['second_derivative']:
                self.compute_second_derivative = True

        self.retain_graph = self.compute_second_derivative
        self.create_graph = self.compute_second_derivative

        # === force to predit ===
        if target & {"forces", "hessians"}:
            self.flags.compute_forces = True
        if self.enable_electric_field:
            if target & {"polarization", "conservative_dipole", "conservative_polarizability", "born_effective_charges"}:
                self.flags.compute_polarization = True
            if target & {"magnetization", "magnetic_susceptibility"}:
                self.flags.compute_magnetization = True

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Optional[Tensor]]:
        # === pre processing ===
        graph = self.prepare_graph(data)

        # === predict ===
        RESULTS = self.readout_fn(data, graph)
        FIRST = self.first_derivative_fn(data, graph, RESULTS)
        SECOND = self.second_derivative_fn(data, graph, RESULTS, FIRST)

        return {
            "energy": RESULTS["energy"],
            "node_energy": RESULTS['node_energy'],
            "charges": RESULTS["charges"],
            "forces": FIRST["forces"],
            "edge_forces": FIRST["edge_forces"],
            "magnetic_forces_0": FIRST["magnetic_forces_0"],
            "magnetic_forces_1": FIRST["magnetic_forces_1"],
            "virials": FIRST["virials"],
            "stress": FIRST["stress"],
            "atomic_virials":FIRST["atomic_virials"],
            "atomic_stresses": FIRST["atomic_stresses"],
            "direct_dipole": RESULTS['direct_dipole'],
            "conservative_dipole": FIRST["polarization"],
            "polarization": FIRST["polarization"],
            "magnetization": FIRST["magnetization"],
            "polarizability": SECOND["polarizability"],
            "direct_polarizability": RESULTS["direct_polarizability"],
            "born_effective_charges": SECOND["born_effective_charges"],
            "hessians": SECOND["hessians"],
            "magnetic_susceptibility": SECOND["magnetic_susceptibility"],
            "les_latent_charges": RESULTS["les_latent_charges"],
            "les_born_effective_charges": RESULTS["les_born_effective_charges"],
            "descriptors": RESULTS['descriptors'],
            "direct_forces": RESULTS["direct_forces"],
            "direct_virials": RESULTS["direct_virials"],
            "direct_stress": RESULTS["direct_stress"],
            # "nuclear_shielding": RESULTS["nuclear_shielding"],
            # "nuclear_chemical_shift": RESULTS["nuclear_chemical_shift"],
            # "elasticity_tensor": RESULTS["elasticity_tensor"],
            # "hill_bulk_modulus": RESULTS["hill_bulk_modulus"],
            # "hill_shear_modulus": RESULTS["hill_shear_modulus"],
            # "hill_young_modulus": RESULTS["hill_young_modulus"],
        }

    def first_derivative_fn(
        self, data: Dict[str, Tensor], graph: Graph, results: Dict[str, Tensor]
    ) -> Dict[str, Optional[Tensor]]:

        E = results["energy"]
        F = None
        V = None
        S = None
        P = None
        M = None
        MAG_F_0 = None
        MAG_F_1 = None
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
        if self.flags.compute_magnetization:
            inputs.append(data["magnetic_field"])
        if self.flags.compute_magnetic_forces_0:
            inputs.append(data["magmoms_0"])
        if self.flags.compute_magnetic_forces_1:
            inputs.append(data["magmoms_1"])
        if self.flags.compute_edge_forces:
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
            F = -grads[idx]
            idx += 1
        if self.flags.compute_stress or self.flags.compute_virials:
            V = -grads[idx]
            VOLUME = torch.linalg.det(data["lattice"]).abs().unsqueeze(-1)
            S = -V / VOLUME.view(-1, 1, 1)
            S = torch.where(torch.abs(S) < 1e10, S, torch.zeros_like(S))
            idx += 1
        if self.flags.compute_polarization or self.flags.compute_conservative_dipole:
            P = -grads[idx]
            idx += 1
        if self.flags.compute_magnetization:
            M = -grads[idx]
            idx += 1
        if self.flags.compute_magnetic_forces_0:
            MAG_F_0 = -grads[idx]
            idx += 1
        if self.flags.compute_magnetic_forces_1:
            MAG_F_1 = -grads[idx]
            idx += 1
        if self.flags.compute_edge_forces:
            EDGE_F = grads[idx] # consistency with LAMMPS
            idx += 1
            A_V, A_S = self.compute_atomic_virials_stresses(
                graph,
                EDGE_F,
                data['edge_index'],
                data['node_attrs'],
                data['batch'],
                graph.lattice,
            )
        return {
            "forces": F,
            "virials": V,
            "stress": S,
            "polarization": P,
            "magnetization": M,
            "magnetic_forces_0": MAG_F_0,
            "magnetic_forces_1": MAG_F_1,
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

        POS = graph.positions
        polarization = first_derivative["polarization"]
        magnetization = first_derivative["magnetization"]

        inputs_list: List[Tensor] = []
        if self.flags.compute_born_effective_charges:
            inputs_list.append(data["positions"])
        if self.flags.compute_conservative_polarizability:
            inputs_list.append(data["electric_field"])

        BEC = None
        ALPHA = None
        CHI_M = None
        H = None
        BECList = []
        ALPHAList = []
        CHI_MList = []
        if self.flags.compute_conservative_polarizability or self.flags.compute_born_effective_charges:
            for i in range(3):  # μ = 0,1,2

                polarization_i = polarization.sum(dim=0)[i]  # sum over batch dimension
                grads = torch.autograd.grad(
                    outputs=polarization_i,
                    inputs=inputs_list,
                    retain_graph=(
                        i != 2 
                        or self.training 
                        or self.flags.compute_hessians 
                        or self.flags.compute_magnetization
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
                        POS.shape[0],
                        3,
                        3,
                        device=POS.device,
                        dtype=POS.dtype,
                    )
                if ALPHA is None:
                    ALPHA = torch.zeros_like(data["electric_field"])

                BECList.append(BEC)  # [atoms, 3]
                ALPHAList.append(ALPHA)  # [B,3]

            BEC = torch.stack(BECList, dim=0)  # [3, atoms, 3]
            BEC = BEC.transpose(1, 0)  #        # [atoms, 3 (pol), 3 (pos)]
            ALPHA = torch.stack(ALPHAList, dim=1)  # [B,3,3]

        if self.flags.compute_magnetization:
            MF = data["magnetic_field"]
            for i in range(3):
                mag_i = magnetization.sum(dim=0)[i]
                CHI_M = torch.autograd.grad(
                    outputs=mag_i,
                    inputs=[MF],
                    retain_graph=(
                        i != 2 
                        or self.training 
                        or self.flags.compute_hessians 
                    ),
                    create_graph=self.training,
                    allow_unused=True,
                )[0]
                if CHI_M is None:
                    CHI_M = torch.zeros_like(MF)
                CHI_MList.append(CHI_M)  # [B,3]
            CHI_M = torch.stack(CHI_MList, dim=1)  # [B,3,3]

        if self.flags.compute_hessians:
            H = self.H_FN(first_derivative["forces"], data, data["ptr"], self.training)
            if H is None:
                H = torch.zeros(
                    POS.shape[0] * POS.shape[0],
                    3,
                    3,
                    device=POS.device,
                    dtype=POS.dtype,
                )

        return {
            "hessians": H,
            "polarizability": ALPHA,
            "born_effective_charges": BEC,
            "magnetic_susceptibility": CHI_M,
        }

    def compute_symmetric_displacement(self, data: Dict[str, Tensor], num_graphs: int) -> Tensor:

        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        displacement.requires_grad_(True)
        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))

        positions = data["positions"]
        positions.requires_grad_(True)
        if data["lattice"] is None:
            data["lattice"] = torch.zeros(
                num_graphs * 3,
                3,
                dtype=data["positions"].dtype,
                device=data["positions"].device,
            )

        data["positions"] = positions + torch.einsum(
            "be,bec->bc", positions, symmetric_displacement[data["batch"]]
        )

        lattice = data["lattice"]
        data["lattice"] = lattice + torch.matmul(lattice, symmetric_displacement)

        return displacement

    def compute_atomic_virials_stresses(
        self,
        graph: Graph,
        edge_forces: Tensor,
        edge_index: Tensor, 
        node_attrs: Tensor,
        batch: Tensor,
        lattice: Tensor, 
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        atomic_virials = None
        atomic_stresses = None

        if self.flags.compute_atomic_virials:
            edge_virials = torch.einsum("zi,zj->zij", edge_forces, graph.edge_vector)
            atomic_virials_source = scatter_sum(
                source=edge_virials, index=edge_index[0], dim=0, dim_size=node_attrs.size(0)
            )
            atomic_virials_target = scatter_sum(
                source=edge_virials, index=edge_index[1], dim=0, dim_size=node_attrs.size(0)
            )
            atomic_virials = (atomic_virials_source + atomic_virials_target) / 2
            atomic_virials = -1 * (atomic_virials + atomic_virials.transpose(-1, -2)) / 2

        if self.flags.compute_atomic_stresses:
            volume = torch.linalg.det(lattice).abs().unsqueeze(-1)
            atomic_stresses = -1* atomic_virials / volume[batch].view(-1, 1, 1)
            atomic_stresses = torch.where(
                torch.abs(atomic_stresses) < 1e10, atomic_stresses, torch.zeros_like(atomic_stresses)
            )

        return atomic_virials, atomic_stresses
    
    def prepare_graph(self, data: Dict[str, torch.Tensor]) -> Graph:

        node_level = (
            data['level'][data['batch']]
            if "level" in data
            else torch.zeros_like(data['batch'], dtype=torch.int64)
        ) # multi-fidelity and multi-head

        if self.lmp:
            for p in self.target_property:
                for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                    if p != 'forces':
                        data[requires_grad_p].requires_grad_(True)
            dtype = data["node_attrs"].dtype
            device =  data["node_attrs"].device 
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
            num_atoms_arange = torch.arange(nlocal, device=positions.device, dtype=torch.int64)
        else:
            for p in self.target_property:
                for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                    data[requires_grad_p].requires_grad_(True)
            dtype = data["node_attrs"].dtype
            device =  data["node_attrs"].device 
            positions = data["positions"]
            num_graphs = data["ptr"].numel() - 1
            displacement = torch.zeros(
                (num_graphs, 3, 3), dtype=dtype, device=device
            )
            if self.flags.compute_virials or self.flags.compute_stress:
                displacement = self.compute_symmetric_displacement(data, num_graphs)
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
            edge_length = (edge_vector**2).sum(dim=1, keepdim=True).sqrt() + 1e-9
            lattice = data['lattice']
            lmp_data = None
            lmp_natoms = (positions.size(0), 0)
            num_atoms_arange = torch.arange(positions.shape[0], device=positions.device, dtype=torch.int64)

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
            node_level=node_level,
            num_atoms_arange=num_atoms_arange,
        )
    
    def compute_hessians_loop(self, forces: Tensor, positions: Tensor) -> Tensor:
        """from MACE"""
        hessian = []
        for grad_elem in forces.view(-1):
            hess_row = torch.autograd.grad(
                outputs=[-1 * grad_elem],
                inputs=[positions],
                grad_outputs=torch.ones_like(grad_elem),
                retain_graph=True,
                create_graph=self.trainging,
                allow_unused=False,
            )[0]
            hess_row = hess_row.detach()
            if hess_row is None:
                hessian.append(torch.zeros_like(positions))
            else:
                hessian.append(hess_row)
        hessian = torch.stack(hessian)
        return hessian


    # @torch.jit.unused
    # def hessians_fn(
    #     forces: Tensor,
    #     positions: Tensor,
    # ) -> Tensor:
    #     """from MACE"""
    #     forces_flatten = forces.view(-1)
    #     numel = forces_flatten.shape[0]

    #     def get_vjp(v):
    #         return torch.autograd.grad(
    #             -1 * forces_flatten,
    #             positions,
    #             v,
    #             retain_graph=True,
    #             create_graph=False,
    #             allow_unused=False,
    #         )

    #     I_N = torch.eye(numel).to(forces.device)
    #     try:
    #         chunk_size = 1 if numel < 64 else 16
    #         gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
    #             I_N
    #         )[0]
    #     except RuntimeError:
    #         gradient = compute_hessians_loop(forces, positions)
    #     if gradient is None:
    #         return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    #     return gradient



    def H_FN(
        self,
        forces: Tensor,
        positions: Tensor,
        PTR: Tensor,
    ) -> Tensor:
        forces_flatten = forces.view(-1)
        numel = forces_flatten.shape[0]

        def get_vjp(v):
            return torch.autograd.grad(
                -1 * forces_flatten,
                positions,
                v,
                retain_graph=True,
                create_graph=self.training,
                allow_unused=False,
            )

        I_N = torch.eye(numel).to(forces.device)
        try:
            chunk_size = 1 if numel < 64 else 16
            gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
                I_N
            )[0]
        except RuntimeError:
            gradient = self.compute_hessians_loop(forces, positions)
        if gradient is None:
            gradient = torch.zeros((forces.shape[0], positions.shape[0], 3, 3))
        N = gradient.shape[1]
        hessian = gradient.view(N, 3, N, 3).permute(0, 2, 1, 3).contiguous()
        blocks = []
        for i in range(len(PTR) - 1):
            start = PTR[i]
            end = PTR[i + 1]
            N_i = end - start
            block = hessian[start:end, start:end, :, :]
            block = block.reshape(N_i * N_i, 3, 3)
            blocks.append(block)
        return torch.cat(blocks, dim=0)



