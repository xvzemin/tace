################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from tace.utils.torch_scatter import scatter_sum


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


def has_valid_labels(label: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    """Return whether any rank has a positive-weight label."""
    weight = label["entropy"].reshape(-1) * label[f"{key}_weight"].reshape(-1)
    present = (weight > 0).any().long()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
    return present


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


def tensor_loss(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    key: str,
    loss: str,
    *,
    voigt: bool = False,
    per_atom: bool = False,
    huber_delta: float = 0.01,
) -> torch.Tensor:
    """Reduce a stress or virial loss over valid labels across all ranks.

    Sample weights multiply the loss without changing the Huber threshold.
    Elementwise losses count tensor entries; L2 losses count structures.
    """
    target = label[key]
    weight = (label["entropy"] * label[f"{key}_weight"]).reshape(-1, 1, 1)
    mask = (weight > 0).expand_as(target)
    # Mask the operands before subtraction, powers or norms, not only the loss.
    error = torch.where(mask, pred[key], 0) - torch.where(mask, target, 0)
    if per_atom:
        error = error / num_atoms_per_graph(label).reshape(-1, 1, 1)
    if voigt:
        error, mask = voigt6_stress(error), voigt6_stress(mask)
        weight = weight.reshape(-1, 1)

    if loss == "mse":
        values = error.square()
    elif loss == "mae":
        values = error.abs()
    elif loss == "huber":
        values = torch.nn.functional.huber_loss(
            error, torch.zeros_like(error), reduction="none", delta=huber_delta
        )
    elif loss == "l2mae":
        values = torch.linalg.vector_norm(error.flatten(1), dim=-1)
        mask = mask.flatten(1).any(dim=-1)
        weight = weight.reshape(-1)
    else:
        raise ValueError(f"Unknown tensor loss: {loss}")

    count = mask.sum()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return (values * weight).sum() * world_size / count.clamp_min(1)


def hessian_loss(pred, label, key, loss, reduction="structure", huber_delta=0.01):
    """Reduce Cartesian-entry errors or atom-pair Frobenius norms.

    Parameters
    ----------
    pred, label : dict
        Predictions and labels in packed ``(sum_g N_g**2, 3, 3)`` layout.
    key : str
        Hessian property name.
    loss : {'mse', 'mae', 'l2mae', 'huber'}
        Elementwise loss, or a Frobenius norm per atom pair for ``l2mae``.
    reduction : {'structure', 'component'}
        Average within each structure, then over structures, or directly over
        valid Cartesian entries. For ``l2mae``, count atom pairs, not entries.
    huber_delta : float
        Huber threshold in the units of the Hessian labels.
    """
    target = label[key]
    if pred[key].shape != target.shape:
        raise ValueError("Hessian predictions and labels must have the same shape.")
    counts = label["ptr"][1:] - label["ptr"][:-1]
    graph = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts.square()
    )
    weight = (label["entropy"] * label[f"{key}_weight"]).reshape(-1)
    mask = (weight[graph, None, None] > 0).expand_as(target)
    error = torch.where(mask, pred[key], 0) - torch.where(mask, target, 0)
    if loss == "mse":
        values = error.square()
    elif loss == "mae":
        values = error.abs()
    elif loss == "l2mae":
        values = torch.linalg.vector_norm(error.flatten(1), dim=-1, keepdim=True)
        mask = mask.flatten(1).any(dim=-1, keepdim=True)
    elif loss == "huber":
        values = torch.nn.functional.huber_loss(
            error, torch.zeros_like(error), reduction="none", delta=huber_delta
        )
    else:
        raise ValueError(f"Unknown Hessian loss: {loss}")
    totals = scatter_sum(
        values.flatten(1).sum(-1), graph, dim=0, dim_size=counts.numel()
    )
    valid = scatter_sum(mask.flatten(1).sum(-1), graph, dim=0, dim_size=counts.numel())
    if reduction == "structure":
        numerator = (totals / valid.clamp_min(1) * weight).sum()
        denominator = (valid > 0).sum()
    elif reduction == "component":
        numerator = (totals * weight).sum()
        denominator = valid.sum()
    else:
        raise ValueError("reduction must be 'structure' or 'component'.")
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    return numerator * world_size / denominator.clamp_min(1)
