################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import copy
from typing import Callable, Dict, List, Tuple

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

to_irrep = {
    "energy": ["0e"],
    "charges": ["0e"],
    "direct_forces": ["1o"],
    "direct_stress": ["0e", "2e"],
    "direct_virials": ["0e", "2e"],
    "direct_dipole": ["1o"],
    "direct_polarizability": ["0e", "2e"],
    "direct_diagonal_hessian": ["0e", "2e"],
    "abs_final_collinear_magmoms": ["0e"],  # TODO, check
}


def get_target_irreps(target_property: List[str]) -> o3.Irreps:
    target_irreps: List[str] = ["0e"]
    for p in target_property:
        target_irreps.extend(to_irrep.get(p, []))
    # return o3.Irreps(list(set(target_irreps)))
    return list(set(target_irreps))


def expand_dims_to(T: torch.Tensor, n_dim: int, dim: int = -1) -> torch.Tensor:
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T


def compute_fixed_charge_dipole(
    charges: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(-1) * 4.8032047  # e·Å to Debye
    return scatter_sum(src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs)


def compute_symmetric_displacement(
    data: Dict[str, torch.Tensor], num_graphs: int
) -> torch.Tensor:

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
    edge_vector: torch.Tensor,
    edge_forces: torch.Tensor,
    edge_index: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    num_nodes: torch.Tensor,
    compute_atomic_virials: bool,
    compute_atomic_stresses: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:

    atomic_virials = None
    atomic_stresses = None

    if compute_atomic_virials or compute_atomic_stresses:
        edge_virials = torch.einsum("zi,zj->zij", edge_forces, edge_vector)
        atomic_virials_source = scatter_sum(
            edge_virials, edge_index[0], dim=0, dim_size=num_nodes
        )
        atomic_virials_target = scatter_sum(
            edge_virials, edge_index[1], dim=0, dim_size=num_nodes
        )
        atomic_virials = (atomic_virials_source + atomic_virials_target) / 2
        atomic_virials = -1 * (atomic_virials + atomic_virials.transpose(-1, -2)) / 2

        volume = torch.linalg.det(lattice).abs().unsqueeze(-1)
        atomic_stresses = -1 * atomic_virials / volume[batch].view(-1, 1, 1)
        atomic_stresses = torch.where(
            torch.abs(atomic_stresses) < 1e10,
            atomic_stresses,
            torch.zeros_like(atomic_stresses),
        )

    return atomic_virials, atomic_stresses


def compute_hessians_vmap(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0]
    except RuntimeError:
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    return gradient


def compute_hessians_loop(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian


def sample_force_components(
    n_atoms: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    k = min(num_samples, n_atoms)
    atom_idx = torch.randperm(n_atoms, device=device)[:k]
    xyz_idx = torch.randint(0, 3, (k,), device=device)
    return torch.stack([atom_idx, xyz_idx], dim=1)


def build_grad_outputs(
    samples_per_graph: List[torch.Tensor],
    ptr: torch.Tensor,
    total_atoms: int,
    device: torch.device,
) -> torch.Tensor:

    all_samples = []

    # Convert local atom indices to global (batch-fidelity_idx) indices
    for graph_id, samples in enumerate(samples_per_graph):
        offset = ptr[graph_id]
        s = samples.clone()
        s[:, 0] += offset  # local -> global atom index
        all_samples.append(s)

    all_samples = torch.cat(all_samples, dim=0)  # 1D, effective global atoms
    K_total = all_samples.shape[0]

    grad_outputs = torch.zeros(
        (K_total, total_atoms, 3),
        device=device,
        dtype=all_samples.dtype,
    )  # TODO, dtype may have bug

    grad_outputs[
        torch.arange(K_total, device=device),
        all_samples[:, 0],
        all_samples[:, 1],
    ] = 1.0

    return grad_outputs


def compute_force_jacobian(
    forces: torch.Tensor,
    positions: torch.Tensor,
    grad_outputs: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    def single_grad(go):
        return torch.autograd.grad(
            outputs=forces,
            inputs=positions,
            grad_outputs=go,
            retain_graph=True,
            create_graph=create_graph,
        )[0]

    return torch.vmap(single_grad)(grad_outputs)


def split_jacobian_per_graph(
    jac: torch.Tensor,
    samples_per_graph: List[torch.Tensor],
    ptr: torch.Tensor,
) -> List[torch.Tensor]:
    jacs_per_graph = []
    row_offset = 0

    for graph_id, samples in enumerate(samples_per_graph):
        k_g = samples.shape[0]
        start, end = ptr[graph_id], ptr[graph_id + 1]

        jac_graph = jac[row_offset : row_offset + k_g, start:end, :]
        jacs_per_graph.append(jac_graph)

        row_offset += k_g

    return jacs_per_graph


def sample_force_jacobian(
    forces: torch.Tensor,
    positions: torch.Tensor,
    ptr: torch.Tensor,
    num_samples: int = 2,
    create_graph: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    device = forces.device
    num_graphs = ptr.numel() - 1
    total_atoms = forces.shape[0]

    samples_per_graph = []
    for g in range(num_graphs):
        n_atoms = (ptr[g + 1] - ptr[g]).item()
        samples = sample_force_components(
            n_atoms=n_atoms,
            num_samples=num_samples,
            device=device,
        )  # [num_samples, 2]
        samples_per_graph.append(samples)

    grad_outputs = build_grad_outputs(
        samples_per_graph=samples_per_graph,
        ptr=ptr,
        total_atoms=total_atoms,
        device=device,
    )

    jac = compute_force_jacobian(
        forces=forces,
        positions=positions,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
    )

    # 4. Split back per graph
    jacs_per_graph = split_jacobian_per_graph(
        jac=jac,
        samples_per_graph=samples_per_graph,
        ptr=ptr,
    )

    return jacs_per_graph, samples_per_graph


def replace_module_recursively(
    model: torch.nn.Module,
    target_cls: type,
    factory: Callable[[torch.nn.Module], torch.nn.Module],
) -> torch.nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, target_cls):
            model._modules[name] = factory(child)
        else:
            replace_module_recursively(child, target_cls, factory)
    return model

def repr_without(module: torch.nn.Module, *names: str) -> str:
    visible = copy.copy(module)
    visible._modules = {
        name: child for name, child in module._modules.items() if name not in names
    }
    return torch.nn.Module.__repr__(visible)