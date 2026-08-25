################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Optional

import torch


def apply_element_weights(
    total_weight: torch.Tensor,
    label: Dict[str, torch.Tensor],
    element_weights: Optional[List[float]],
) -> torch.Tensor:
    if element_weights is None:
        return total_weight

    node_attrs = label["node_attrs"]
    # if node_attrs.ndim != 2:
    #     raise ValueError("node_attrs must have shape (num_atoms, num_elements).")
    if len(element_weights) != node_attrs.shape[-1]:
        raise ValueError(
            "element_weights must provide one weight for every element in "
            f"node_attrs; expected {node_attrs.shape[-1]}, got "
            f"{len(element_weights)}."
        )
    weights = torch.as_tensor(
        list(element_weights),
        dtype=total_weight.dtype,
        device=total_weight.device,
    )
    atom_weights = node_attrs @ weights
    shape = (atom_weights.shape[0],) + (1,) * (total_weight.ndim - 1)
    return total_weight * atom_weights.reshape(shape)


def num_atoms_per_graph(label: Dict[str, torch.Tensor]) -> torch.Tensor:
    return label["ptr"][1:] - label["ptr"][:-1]


def polarization_error_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    key: str = "polarization",
) -> torch.Tensor:
    lattice = label["lattice"]
    num_atoms = num_atoms_per_graph(label).reshape(-1, 1)
    error = pred[key] - label[key]
    error = torch.einsum("bi, bij -> bj", error, torch.linalg.inv(lattice))
    error = torch.remainder(error, 1.0)
    error = torch.where(error > 0.5, error - 1.0, error)
    error = torch.where(error < -0.5, error + 1.0, error)
    error = torch.einsum("bi, bij -> bj", error, lattice)
    return error / num_atoms


def voigt6_stress(stress: torch.Tensor) -> torch.Tensor:
    return stress.reshape(-1, 9)[:, [0, 4, 8, 5, 2, 1]]
