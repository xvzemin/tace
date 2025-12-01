################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Callable


import torch
from torch import Tensor

LOSS_FN: Dict[str, Callable] = {}


def register_loss(func: Callable = None, *, key: str = None):
    def decorator(f: Callable):
        k = key if key is not None else f.__name__
        LOSS_FN[k] = f
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)


@register_loss
def mse_energy(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = label.entropy * label.energy_weight
    return torch.mean(torch.square((label["energy"] - pred["energy"])) * total_weight)


@register_loss
def mse_energy_per_atom(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = label.entropy * label.energy_weight
    return torch.mean(
        torch.square(
            (label["energy"] - pred["energy"]) / (label.ptr[1:] - label.ptr[:-1])
        )
        * total_weight
    )


@register_loss
def mae_energy_per_atom(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = label.entropy * label.energy_weight
    return torch.mean(
        torch.abs((label["energy"] - pred["energy"]) / (label.ptr[1:] - label.ptr[:-1]))
        * total_weight
    )


@register_loss
def mse_forces(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.forces_weight[
        batch
    ].unsqueeze(-1)
    return torch.mean(torch.square(pred["forces"] - label["forces"]) * total_weight)


@register_loss
def mean_normed_forces(pred, label) -> Tensor:
    return torch.mean(
        torch.linalg.vector_norm(pred["forces"] - label["forces"], ord=2, dim=-1)
    )


@register_loss
def conditional_huber_forces(
    pred_forces: Tensor, label_forces: Tensor, huber_delta: float
) -> Tensor:
    "From MACE https://github.com/ACEsuit/mace"
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=label_forces.device, dtype=label_forces.dtype
    )
    norm_forces = torch.norm(label_forces, dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    c4 = ~(c1 | c2 | c3)
    se = torch.zeros_like(pred_forces)
    se[c1] = torch.nn.functional.huber_loss(
        label_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        label_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        label_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        label_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    return torch.mean(se)


@register_loss
def mse_stress(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = label.entropy * label.stress_weight
    return torch.mean(
        torch.square(pred["stress"] - label["stress"])
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )


@register_loss
def mse_virials_per_atom(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = label.entropy * label.virials_weight
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).view(-1, 1, 1)
    return torch.mean(
        (torch.square(pred["virials"] - label["virials"]) / num_atoms)
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )


@register_loss
def mse_direct_dipole(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = (label.entropy * label.direct_dipole_weight).unsqueeze(-1)
    return torch.mean(torch.square((label["direct_dipole"] - pred["direct_dipole"])) * total_weight)


@register_loss
def mse_direct_dipole_per_atom(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = (label.entropy * label.direct_dipole_weight).unsqueeze(-1)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1)
    return torch.mean(
        torch.square((label["direct_dipole"] - pred["direct_dipole"]) / num_atoms) * total_weight
    )

@register_loss
def mse_conservative_dipole(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = (label.entropy * label.conservative_dipole_weight).unsqueeze(-1)
    return torch.mean(
        torch.square(label["conservative_dipole"] - pred["conservative_dipole"]) * total_weight
    )


@register_loss
def mse_conservative_dipole_per_atom(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = (label.entropy * label.conservative_dipole_weight).unsqueeze(-1)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1)
    return torch.mean(
        torch.square((label["conservative_dipole"] - pred["conservative_dipole"]) / num_atoms) * total_weight
    )

@register_loss
def mse_direct_polarizability(
    pred: Dict[str, Tensor], label: Dict[str, Tensor]
) -> Tensor:
    total_weight = (
        (label.entropy * label.direct_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    return torch.mean(
        torch.square(label["direct_polarizability"] - pred["direct_polarizability"])
        * total_weight
    )

@register_loss
def mse_direct_polarizability_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor]
) -> Tensor:
    total_weight = (
        (label.entropy * label.direct_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
    return torch.mean(
        torch.square((label["direct_polarizability"] - pred["direct_polarizability"]) / num_atoms)
        * total_weight
    )

@register_loss
def mse_conservative_polarizability(
    pred: Dict[str, Tensor], label: Dict[str, Tensor]
) -> Tensor:
    total_weight = (
        (label.entropy * label.conservative_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    return torch.mean(
        torch.square(label["conservative_polarizability"] - pred["conservative_polarizability"])
        * total_weight
    )

@register_loss
def mse_conservative_polarizability_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor]
) -> Tensor:
    total_weight = (
        (label.entropy * label.conservative_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
    return torch.mean(
        torch.square((label["conservative_polarizability"] - pred["conservative_polarizability"]) / num_atoms)
        * total_weight
    )

@register_loss
def mse_hessians(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.hessians_weight)[batch][batch]  # TODO check
    total_weight = total_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return torch.mean(
        torch.square((label["hessians"] - pred["hessians"]))
        * total_weight  # hessians shape: [atoms*atom, 3, 3]
    )


@register_loss
def mse_born_effective_charges(
    pred: Dict[str, Tensor], label: Dict[str, Tensor]
) -> Tensor:
    batch = label.batch
    total_weight = (
        (label.entropy * label.born_effective_charges_weight)[batch]
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

    return torch.mean(
        torch.square(label["born_effective_charges"] - pred["born_effective_charges"])
        * total_weight
    )


@register_loss
def mse_polarization_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor]
) -> Tensor:
    ptr = label["ptr"]
    lattice = label["lattice"]
    num_atoms = (ptr[1:] - ptr[:-1]).reshape(-1, 1)
    error = pred["polarization"] - label["polarization"]
    error = torch.einsum("bi, bij -> bj", error, torch.linalg.inv(lattice))
    error = torch.remainder(error, 1.0)
    error = torch.where(error > 0.5, error - 1.0, error)
    error = torch.where(error < -0.5, error + 1.0, error)
    error = torch.einsum("bi, bij -> bj", error, lattice)
    return torch.mean(torch.square(error / num_atoms))


@register_loss
def mse_charges(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.charges_weight)[batch]
    return torch.mean(torch.square(pred["charges"] - label["charges"]) * total_weight)

# # no use
@register_loss
def mse_magmoms(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.magmoms_weight[
        batch
    ].unsqueeze(-1)
    return torch.mean(torch.square(pred["magmoms"] - label["magmoms"]) * total_weight)

@register_loss
def mse_magnetic_forces_0(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.magnetic_forces_0_weight)[batch]
    return torch.mean(torch.square(pred["magnetic_forces_0"] - label["magnetic_forces_0"]) * total_weight)

@register_loss
def mse_magnetic_forces_1(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.magnetic_forces_1_weight[
        batch
    ].unsqueeze(-1)
    return torch.mean(
        torch.square(pred["magnetic_forces_1"] - label["magnetic_forces_1"]) * total_weight
    )


@register_loss
def mse_direct_forces(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.direct_forces_weight[
        batch
    ].unsqueeze(-1)
    return torch.mean(torch.square(pred["direct_forces"] - label["direct_forces"]) * total_weight)


@register_loss
def mse_direct_stress(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = label.entropy * label.direct_stress_weight
    return torch.mean(
        torch.square(pred["direct_stress"] - label["direct_stress"])
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )

@register_loss
def mse_direct_virials_per_atom(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = label.entropy * label.direct_virials_weight
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).view(-1, 1, 1)
    return torch.mean(
        torch.square((pred["direct_virials"] - label["direct_virials"]) / num_atoms)
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )

@register_loss
def mse_nuclear_chemical_shift(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.nuclear_chemical_shift_weight)[batch]
    return torch.mean(torch.square(pred["nuclear_chemical_shift"] - label["nuclear_chemical_shift"]) * total_weight)

@register_loss
def mse_nuclear_shielding(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.nuclear_shielding_weight)[batch]
    return torch.mean(
        torch.square(
            pred["nuclear_shielding"] - label["nuclear_shielding"]
        ) * total_weight.unsqueeze(-1).unsqueeze(-1)
    )

@register_loss
def mse_elasticity_tensor(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = (label.entropy * label.elasticity_tensor_weight)
    return torch.mean(
        torch.square(
            pred["elasticity_tensor"] - label["elasticity_tensor"]
        ) * total_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    )

@register_loss
def mse_elasticity_tensor(pred: Dict[str, Tensor], label: Dict[str, Tensor]) -> Tensor:
    total_weight = (label.entropy * label.elasticity_tensor_weight)
    return torch.mean(
        torch.square(
            pred["elasticity_tensor"] - label["elasticity_tensor"]
        ) * total_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    )