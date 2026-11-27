################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from tace.utils.precision import DATA_DTYPE


def loss_inputs(pred, label):
    """Convert floating loss operands to FP64 without modifying the input mappings."""
    pred = {
        key: value.to(DATA_DTYPE)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for key, value in pred.items()
    }
    label = {
        key: value.to(DATA_DTYPE)
        if isinstance(value, torch.Tensor)
        and value.is_floating_point()
        # Geometric regularizers differentiate with respect to the original coordinates.
        and key != "positions"
        else value
        for key, value in label.items()
    }
    return pred, label


def apply_element_weights(
    total_weight: torch.Tensor,
    label: Dict[str, torch.Tensor],
    element_weights: Optional[List[float] | torch.Tensor],
) -> torch.Tensor:
    if element_weights is None:
        return total_weight

    # The table is a loss constant, not a trainable operand. A prepositioned
    # tensor must not be unpacked into Python scalars during CUDA Graph capture.
    values = (
        element_weights.detach()
        if torch.is_tensor(element_weights) and element_weights.ndim == 1
        else list(element_weights)
    )
    weights = torch.as_tensor(
        values,
        dtype=total_weight.dtype,
        device=total_weight.device,
    )
    atom_weights = weights[label["node_type"]]
    shape = (atom_weights.shape[0],) + (1,) * (total_weight.ndim - 1)
    return total_weight * atom_weights.reshape(shape)


def num_atoms_per_graph(label: Dict[str, torch.Tensor]) -> torch.Tensor:
    return label["ptr"][1:] - label["ptr"][:-1]


def has_valid_labels(label: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    """Return whether any rank has a positive-weight label."""
    if f"_valid_{key}_count" in label:
        return (label[f"_valid_{key}_count"] > 0).long()
    weight = label["entropy"].to(DATA_DTYPE).reshape(-1) * label[f"{key}_weight"].to(
        DATA_DTYPE
    ).reshape(-1)
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
    # xx, yy, zz, yz, xz, xy without an implicit CPU index tensor in capture.
    columns = stress.reshape(-1, 9).unbind(1)
    return torch.stack([columns[i] for i in (0, 4, 8, 5, 2, 1)], dim=1)


def prepare_tensor_loss_counts(label, properties):
    """Reduce valid structure counts before entering a compiled training graph."""
    label = dict(label.items())
    keys = tuple(dict.fromkeys(properties))
    if not keys:
        return label
    names, values = [], []
    for key in keys:
        valid = (
            label["entropy"].to(DATA_DTYPE).reshape(-1)
            * label[f"{key}_weight"].to(DATA_DTYPE).reshape(-1)
            > 0
        )
        names.append(f"_valid_{key}_count")
        values.append(valid.sum())
        if key == "direct_hessian":
            names.append(f"_valid_{key}_pair_count")
            values.append((valid * num_atoms_per_graph(label).square()).sum())
    counts = values[0].view(1) if len(values) == 1 else torch.stack(values)
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    label.update(zip(names, counts))
    label["_loss_world_size"] = counts.new_full((), world_size)
    return label


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
    valid = weight > 0
    # Mask the operands before subtraction, powers or norms, not only the loss.
    error = torch.where(valid, pred[key], 0) - torch.where(valid, target, 0)
    if per_atom:
        error = error / num_atoms_per_graph(label).reshape(-1, 1, 1)
    if voigt:
        error = voigt6_stress(error)
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
        weight = weight.reshape(-1)
    else:
        raise ValueError(f"Unknown tensor loss: {loss}")

    count_key = f"_valid_{key}_count"
    prepared = count_key in label
    if loss == "l2mae":
        count = label[count_key] if prepared else valid.sum()
    else:
        components = 6 if voigt else 9
        # Promote before expanding so the reduction does not allocate G*components integers.
        count = (
            label[count_key] * components
            if prepared
            else valid.long().expand(-1, components, 1).sum()
        )
    world_size = label["_loss_world_size"] if prepared else 1
    if not prepared and dist.is_available() and dist.is_initialized():
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
    pairs = num_atoms_per_graph(label).square()
    graph = torch.repeat_interleave(
        torch.arange(pairs.numel(), device=pairs.device),
        pairs,
        output_size=target.shape[0],
    )
    weight = (label["entropy"] * label[f"{key}_weight"]).reshape(-1)
    valid = (weight > 0) & (pairs > 0)
    error = torch.where(valid[graph, None, None], pred[key] - target, 0)
    if loss == "mse":
        values = error.square()
    elif loss == "mae":
        values = error.abs()
    elif loss == "l2mae":
        values = torch.linalg.vector_norm(error.flatten(1), dim=-1)
    elif loss == "huber":
        values = torch.nn.functional.huber_loss(
            error, torch.zeros_like(error), reduction="none", delta=huber_delta
        )
    else:
        raise ValueError(f"Unknown Hessian loss: {loss}")
    components = 1 if loss == "l2mae" else 9
    if components != 1:
        values = values.sum(dim=(-2, -1))
    if reduction == "structure":
        weight = weight.to(torch.promote_types(values.dtype, weight.dtype))
        weight = weight / (pairs * components).clamp_min(1)
    elif reduction != "component":
        raise ValueError("reduction must be 'structure' or 'component'.")
    numerator = (values * weight[graph]).sum()
    if f"_valid_{key}_count" in label:
        world_size = label["_loss_world_size"]
        denominator = (
            label[f"_valid_{key}_count"]
            if reduction == "structure"
            else label[f"_valid_{key}_pair_count"] * components
        )
    else:
        denominator = (
            valid.sum()
            if reduction == "structure"
            else (pairs * valid).sum() * components
        )
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    return numerator * world_size / denominator.clamp_min(1)
