################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Optional

import torch

from .common import apply_element_weights, polarization_error_per_atom, voigt6_stress
from .mse_fn import register_loss


def _per_atom_huber_loss(
    pred: torch.Tensor,
    label_value: torch.Tensor,
    total_weight: torch.Tensor,
    label: Dict[str, torch.Tensor],
    huber_delta: float,
    element_weights: Optional[List[float]],
) -> torch.Tensor:
    loss = torch.nn.functional.huber_loss(
        total_weight * label_value,
        total_weight * pred,
        reduction="none",
        delta=huber_delta,
    )
    return torch.mean(apply_element_weights(loss, label, element_weights))


@register_loss
def huber_energy(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = label["entropy"] * label["energy_weight"]
    key = "energy"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_energy_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = label["entropy"] * label["energy_weight"]
    num_atoms = label["ptr"][1:] - label["ptr"][:-1]
    key = "energy"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_forces(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    batch = label["batch"]
    total_weight = (label["entropy"] * label["forces_weight"])[batch].unsqueeze(-1)
    key = "forces"
    return _per_atom_huber_loss(
        pred[key], label[key], total_weight, label, huber_delta, element_weights
    )


@register_loss
def huber_stress(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        (label["entropy"] * label["stress_weight"]).unsqueeze(-1).unsqueeze(-1)
    )
    key = "stress"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_voigt_stress(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    key = "stress"
    total_weight = (label["entropy"] * label["stress_weight"]).unsqueeze(-1)
    return torch.nn.functional.huber_loss(
        total_weight * voigt6_stress(label[key]),
        total_weight * voigt6_stress(pred[key]),
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_virials(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        (label["entropy"] * label["virials_weight"]).unsqueeze(-1).unsqueeze(-1)
    )
    key = "virials"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_virials_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        (label["entropy"] * label["virials_weight"]).unsqueeze(-1).unsqueeze(-1)
    )
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).view(-1, 1, 1)
    key = "virials"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_direct_forces(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    batch = label["batch"]
    total_weight = label["entropy"][batch].unsqueeze(-1) * label[
        "direct_forces_weight"
    ][batch].unsqueeze(-1)
    key = "direct_forces"
    return _per_atom_huber_loss(
        pred[key], label[key], total_weight, label, huber_delta, element_weights
    )


@register_loss
def huber_direct_stress(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        (label["entropy"] * label["direct_stress_weight"]).unsqueeze(-1).unsqueeze(-1)
    )
    key = "direct_stress"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_voigt_direct_stress(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    key = "direct_stress"
    total_weight = (label["entropy"] * label["direct_stress_weight"]).unsqueeze(-1)
    return torch.nn.functional.huber_loss(
        total_weight * voigt6_stress(label[key]),
        total_weight * voigt6_stress(pred[key]),
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_direct_virials_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        (label["entropy"] * label["direct_virials_weight"]).unsqueeze(-1).unsqueeze(-1)
    )
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).view(-1, 1, 1)
    key = "direct_virials"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_direct_dipole(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (label["entropy"] * label["direct_dipole_weight"]).unsqueeze(-1)
    key = "direct_dipole"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_direct_dipole_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (label["entropy"] * label["direct_dipole_weight"]).unsqueeze(-1)
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).unsqueeze(-1)
    key = "direct_dipole"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_conservative_dipole(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (label["entropy"] * label["conservative_dipole_weight"]).unsqueeze(
        -1
    )
    key = "conservative_dipole"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_conservative_dipole_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (label["entropy"] * label["conservative_dipole_weight"]).unsqueeze(
        -1
    )
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).unsqueeze(-1)
    key = "conservative_dipole"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_direct_polarizability(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        (label["entropy"] * label["direct_polarizability_weight"])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    key = "direct_polarizability"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_direct_polarizability_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        (label["entropy"] * label["direct_polarizability_weight"])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).unsqueeze(-1).unsqueeze(-1)
    key = "direct_polarizability"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_conservative_polarizability(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        (label["entropy"] * label["conservative_polarizability_weight"])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    key = "conservative_polarizability"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_conservative_polarizability_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        (label["entropy"] * label["conservative_polarizability_weight"])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).unsqueeze(-1).unsqueeze(-1)
    key = "conservative_polarizability"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_born_effective_charges(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    batch = label["batch"]
    total_weight = (
        (label["entropy"] * label["born_effective_charges_weight"])[batch]
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    key = "born_effective_charges"
    return _per_atom_huber_loss(
        pred[key], label[key], total_weight, label, huber_delta, element_weights
    )


@register_loss
def huber_polarization_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    key = "polarization"
    total_weight = (label["entropy"] * label["polarization_weight"]).unsqueeze(-1)
    error = polarization_error_per_atom(pred, label, key)
    return torch.nn.functional.huber_loss(
        total_weight * error,
        torch.zeros_like(error),
        delta=huber_delta,
        reduction="mean",
    )


@register_loss
def huber_charges(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    batch = label["batch"]
    total_weight = (label["entropy"] * label["charges_weight"])[batch]
    key = "charges"
    return _per_atom_huber_loss(
        pred[key], label[key], total_weight, label, huber_delta, element_weights
    )


@register_loss
def huber_final_collinear_magmoms(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    batch = label["batch"]
    total_weight = (label["entropy"] * label["final_collinear_magmoms_weight"])[batch]
    key = "final_collinear_magmoms"
    return _per_atom_huber_loss(
        pred[key], label[key], total_weight, label, huber_delta, element_weights
    )


@register_loss
def huber_abs_final_collinear_magmoms(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    batch = label["batch"]
    total_weight = (label["entropy"] * label["abs_final_collinear_magmoms_weight"])[
        batch
    ]
    key = "abs_final_collinear_magmoms"
    return _per_atom_huber_loss(
        pred[key], label[key], total_weight, label, huber_delta, element_weights
    )


@register_loss
def huber_final_noncollinear_magmoms(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    batch = label["batch"]
    total_weight = label["entropy"][batch].unsqueeze(-1) * label[
        "final_noncollinear_magmoms_weight"
    ][batch].unsqueeze(-1)
    key = "final_noncollinear_magmoms"
    return _per_atom_huber_loss(
        pred[key], label[key], total_weight, label, huber_delta, element_weights
    )


@register_loss
def huber_collinear_magnetic_forces(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    batch = label["batch"]
    total_weight = (label["entropy"] * label["collinear_magnetic_forces_weight"])[batch]
    key = "collinear_magnetic_forces"
    return _per_atom_huber_loss(
        pred[key], label[key], total_weight, label, huber_delta, element_weights
    )


@register_loss
def huber_noncollinear_magnetic_forces(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    batch = label["batch"]
    total_weight = label["entropy"][batch].unsqueeze(-1) * label[
        "noncollinear_magnetic_forces_weight"
    ][batch].unsqueeze(-1)
    key = "noncollinear_magnetic_forces"
    return _per_atom_huber_loss(
        pred[key], label[key], total_weight, label, huber_delta, element_weights
    )


@register_loss
def huber_total_collinear_magmom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = label["entropy"] * label["total_collinear_magmom_weight"]
    key = "total_collinear_magmom"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_total_collinear_magmom_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = label["entropy"] * label["total_collinear_magmom_weight"]
    num_atoms = label["ptr"][1:] - label["ptr"][:-1]
    key = "total_collinear_magmom"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_total_noncollinear_magmom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        label["entropy"] * label["total_noncollinear_magmom_weight"]
    ).unsqueeze(-1)
    key = "total_noncollinear_magmom"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_total_noncollinear_magmom_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    total_weight = (
        label["entropy"] * label["total_noncollinear_magmom_weight"]
    ).unsqueeze(-1)
    num_atoms = (label["ptr"][1:] - label["ptr"][:-1]).unsqueeze(-1)
    key = "total_noncollinear_magmom"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )
