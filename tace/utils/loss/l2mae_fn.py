################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict

import torch

from .common import num_atoms_per_graph, polarization_error_per_atom, voigt6_stress
from .mse_fn import register_loss


@register_loss
def l2mae_energy(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "energy"
    total_weight = label["entropy"] * label["energy_weight"]
    return torch.mean(torch.abs(label[key] - pred[key]) * total_weight)


@register_loss
def l2mae_energy_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "energy"
    total_weight = label["entropy"] * label["energy_weight"]
    num_atoms = num_atoms_per_graph(label)
    return torch.mean(torch.abs((label[key] - pred[key]) / num_atoms) * total_weight)


@register_loss
def l2mae_forces(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "forces"
    batch = label["batch"]
    total_weight = (label["entropy"] * label["forces_weight"])[batch]
    return torch.mean(
        torch.linalg.vector_norm(pred[key] - label[key], ord=2, dim=-1) * total_weight
    )


@register_loss
def l2mae_stress(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "stress"
    total_weight = label["entropy"] * label["stress_weight"]
    return torch.mean(
        torch.linalg.vector_norm(pred[key] - label[key], ord=2, dim=(1, 2))
        * total_weight
    )


@register_loss
def l2mae_voigt_stress(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "stress"
    total_weight = label["entropy"] * label["stress_weight"]
    error = voigt6_stress(pred[key] - label[key])
    return torch.mean(torch.linalg.vector_norm(error, ord=2, dim=-1) * total_weight)


@register_loss
def l2mae_virials(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "virials"
    total_weight = label["entropy"] * label["virials_weight"]
    return torch.mean(
        torch.linalg.vector_norm(pred[key] - label[key], ord=2, dim=(1, 2))
        * total_weight
    )


@register_loss
def l2mae_virials_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "virials"
    total_weight = label["entropy"] * label["virials_weight"]
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).view(-1, 1, 1)
    return torch.mean(
        torch.linalg.vector_norm(
            (pred[key] - label[key]) / num_atoms, ord=2, dim=(1, 2)
        )
        * total_weight
    )


@register_loss
def l2mae_direct_forces(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_forces"
    batch = label["batch"]
    total_weight = (label["entropy"] * label["direct_forces_weight"])[batch]
    return torch.mean(
        torch.linalg.vector_norm(pred[key] - label[key], ord=2, dim=-1) * total_weight
    )


@register_loss
def l2mae_direct_stress(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_stress"
    total_weight = label["entropy"] * label["direct_stress_weight"]
    return torch.mean(
        torch.linalg.vector_norm(pred[key] - label[key], ord=2, dim=(1, 2))
        * total_weight
    )


@register_loss
def l2mae_voigt_direct_stress(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_stress"
    total_weight = label["entropy"] * label["direct_stress_weight"]
    error = voigt6_stress(pred[key] - label[key])
    return torch.mean(torch.linalg.vector_norm(error, ord=2, dim=-1) * total_weight)


@register_loss
def l2mae_direct_virials_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_virials"
    total_weight = label["entropy"] * label["direct_virials_weight"]
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).view(-1, 1, 1)
    return torch.mean(
        torch.linalg.vector_norm(
            (pred[key] - label[key]) / num_atoms,
            ord=2,
            dim=(1, 2),
        )
        * total_weight
    )


@register_loss
def l2mae_direct_dipole(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_dipole"
    total_weight = label["entropy"] * label["direct_dipole_weight"]
    return torch.mean(
        torch.linalg.vector_norm(
            label[key] - pred[key],
            ord=2,
            dim=-1,
        )
        * total_weight
    )


@register_loss
def l2mae_direct_dipole_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_dipole"
    total_weight = label["entropy"] * label["direct_dipole_weight"]
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).view(-1, 1)
    return torch.mean(
        torch.linalg.vector_norm(
            (label[key] - pred[key]) / num_atoms,
            ord=2,
            dim=-1,
        )
        * total_weight
    )


@register_loss
def l2mae_conservative_dipole(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "conservative_dipole"
    total_weight = label["entropy"] * label["conservative_dipole_weight"]
    return torch.mean(
        torch.linalg.vector_norm(
            label[key] - pred[key],
            ord=2,
            dim=-1,
        )
        * total_weight
    )


@register_loss
def l2mae_conservative_dipole_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "conservative_dipole"
    total_weight = label["entropy"] * label["conservative_dipole_weight"]
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).view(-1, 1)
    return torch.mean(
        torch.linalg.vector_norm(
            (label[key] - pred[key]) / num_atoms,
            ord=2,
            dim=-1,
        )
        * total_weight
    )


@register_loss
def l2mae_direct_polarizability(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_polarizability"
    total_weight = label["entropy"] * label["direct_polarizability_weight"]
    return torch.mean(
        torch.linalg.vector_norm(
            label[key] - pred[key],
            ord=2,
            dim=(1, 2),
        )
        * total_weight
    )


@register_loss
def l2mae_direct_polarizability_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_polarizability"
    total_weight = label["entropy"] * label["direct_polarizability_weight"]
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).view(-1, 1, 1)
    return torch.mean(
        torch.linalg.vector_norm(
            (label[key] - pred[key]) / num_atoms,
            ord=2,
            dim=(1, 2),
        )
        * total_weight
    )


@register_loss
def l2mae_conservative_polarizability(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "conservative_polarizability"
    total_weight = label["entropy"] * label["conservative_polarizability_weight"]
    return torch.mean(
        torch.linalg.vector_norm(
            label[key] - pred[key],
            ord=2,
            dim=(1, 2),
        )
        * total_weight
    )


@register_loss
def l2mae_conservative_polarizability_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "conservative_polarizability"
    total_weight = label["entropy"] * label["conservative_polarizability_weight"]
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).view(-1, 1, 1)
    return torch.mean(
        torch.linalg.vector_norm(
            (label[key] - pred[key]) / num_atoms,
            ord=2,
            dim=(1, 2),
        )
        * total_weight
    )


@register_loss
def l2mae_born_effective_charges(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "born_effective_charges"
    batch = label["batch"]
    total_weight = (label["entropy"] * label["born_effective_charges_weight"])[batch]
    return torch.mean(
        torch.linalg.vector_norm(
            label[key] - pred[key],
            ord=2,
            dim=(1, 2),
        )
        * total_weight
    )


@register_loss
def l2mae_polarization_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "polarization"
    total_weight = label["entropy"] * label["polarization_weight"]
    error = polarization_error_per_atom(pred, label, key)
    return torch.mean(torch.linalg.vector_norm(error, ord=2, dim=-1) * total_weight)


@register_loss
def l2mae_charges(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "charges"
    batch = label["batch"]
    total_weight = (label["entropy"] * label["charges_weight"])[batch]
    return torch.mean(torch.abs(pred[key] - label[key]) * total_weight)


@register_loss
def l2mae_final_collinear_magmoms(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "final_collinear_magmoms"
    batch = label["batch"]
    total_weight = (label["entropy"] * label["final_collinear_magmoms_weight"])[batch]
    return torch.mean(torch.abs(pred[key] - label[key]) * total_weight)


@register_loss
def l2mae_abs_final_collinear_magmoms(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "abs_final_collinear_magmoms"
    batch = label["batch"]
    total_weight = (label["entropy"] * label["abs_final_collinear_magmoms_weight"])[
        batch
    ]
    return torch.mean(torch.abs(pred[key] - label[key]) * total_weight)


@register_loss
def l2mae_final_noncollinear_magmoms(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "final_noncollinear_magmoms"
    batch = label["batch"]
    total_weight = (label["entropy"] * label["final_noncollinear_magmoms_weight"])[
        batch
    ]
    return torch.mean(
        torch.linalg.vector_norm(pred[key] - label[key], ord=2, dim=-1) * total_weight
    )


@register_loss
def l2mae_collinear_magnetic_forces(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "collinear_magnetic_forces"
    batch = label["batch"]
    total_weight = (label["entropy"] * label["collinear_magnetic_forces_weight"])[batch]
    return torch.mean(torch.abs(pred[key] - label[key]) * total_weight)


@register_loss
def l2mae_noncollinear_magnetic_forces(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "noncollinear_magnetic_forces"
    batch = label["batch"]
    total_weight = (label["entropy"] * label["noncollinear_magnetic_forces_weight"])[
        batch
    ]
    return torch.mean(
        torch.linalg.vector_norm(pred[key] - label[key], ord=2, dim=-1) * total_weight
    )


@register_loss
def l2mae_total_collinear_magmom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "total_collinear_magmom"
    total_weight = label["entropy"] * label["total_collinear_magmom_weight"]
    return torch.mean(torch.abs(label[key] - pred[key]) * total_weight)


@register_loss
def l2mae_total_collinear_magmom_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "total_collinear_magmom"
    total_weight = label["entropy"] * label["total_collinear_magmom_weight"]
    num_atoms = label["ptr"][1:] - label["ptr"][:-1]
    return torch.mean(torch.abs((label[key] - pred[key]) / num_atoms) * total_weight)


@register_loss
def l2mae_total_noncollinear_magmom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "total_noncollinear_magmom"
    total_weight = label["entropy"] * label["total_noncollinear_magmom_weight"]
    return torch.mean(
        torch.linalg.vector_norm(label[key] - pred[key], ord=2, dim=-1) * total_weight
    )


@register_loss
def l2mae_total_noncollinear_magmom_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "total_noncollinear_magmom"
    total_weight = label["entropy"] * label["total_noncollinear_magmom_weight"]
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).unsqueeze(-1)
    return torch.mean(
        torch.linalg.vector_norm((label[key] - pred[key]) / num_atoms, ord=2, dim=-1)
        * total_weight
    )
