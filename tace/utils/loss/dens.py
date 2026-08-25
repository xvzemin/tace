import dataclasses
from typing import List, Optional, Union

import torch

from ..torch_scatter import scatter
from .common import apply_element_weights, voigt6_stress
from .mse_fn import register_loss


@dataclasses.dataclass
class DenoisingPosParams:
    prob: float = 0.5
    fixed_noise_std: bool = True
    std: float = 0.025
    corrupt_ratio: float = 0.5
    all_atoms: bool = True
    min_num_atoms: Union[int, None] = None
    strict_max_ratio: Union[float, None] = 0.75
    max_force_norm: Union[float, None] = 2.5
    max_stress_norm: Union[float, None] = None
    max_mean_force_norm: Union[float, None] = None


DeNS = DenoisingPosParams()

def add_gaussian_noise_to_position(
    batch,
    prob=DeNS.prob,
    std=DeNS.std,
    corrupt_ratio=DeNS.corrupt_ratio,
    all_atoms=DeNS.all_atoms,
    min_num_atoms=DeNS.min_num_atoms,  # => dens_batch_mask
    strict_max_ratio=DeNS.strict_max_ratio,
    max_forces_norm=DeNS.max_force_norm,
    max_stress_norm=DeNS.max_stress_norm,
    max_mean_forces_norm=DeNS.max_mean_force_norm,
):
    """
    1.  Update `pos` in `batch`.
    2.  Add `noise_vec` to `batch`, which will serve as the target for denoising positions.
    3.  Add `denoising_pos_forward` to switch to denoising mode during training.
    4.  Add `noise_mask` for partially corrupted structures when `corrupt_ratio` is not None.
    5.  If `all_atoms` == True, we add noise to all atoms including fixed ones.
    6.  Check whether `batch` has `skip_dens`. We do not add noise to structures when `skip_dens` == True.
    7.  If `min_num_atoms` != None, we do not add noise to structures with numbers of atoms
        less than `min_num_atoms`.
    8.  Add `dens_batch_mask` to specify which graphs we apply DeNS.
        This is used when `min_num_atoms` is not None or `strict_ratio` == True.
        `dens_batch_mask` is to be used to mask out stress prediction during DeNS.
    9.  If `strict_max_ratio` is not `None`, we skip denoising for a certain structure if the number of
        corrupted atoms is great than `strict_max_ratio` * number of atoms in that structure.
    10. If `max_force_norm` is not `None`, we skip denoising if the maximum of L2 norm of forces is
        greater than `max_force_norm`.
    11. If `max_stress_norm` is not `None`, we do not add noise to atoms with
        stress norm > `max_stress_norm`.
    12. If `max_mean_force_norm` is not `None`, we skip denoising if the L2 norm of the sum of atomwise forces
        is greater than `max_mean_force_norm`.
    """
    B = batch["node_attrs"].size(0)
    num_atoms = batch["ptr"][1:] - batch["ptr"][:-1]
    num_graphs = len(batch["ptr"]) - 1
    dtype = batch["node_attrs"].dtype
    device = batch["node_attrs"].device
    batch_index = batch["batch"]

    if not torch.rand(1).item() < prob:
        batch["noise_vec"] = torch.zeros_like(batch["positions"])
        batch["dens_batch_mask"] = torch.zeros(
            num_graphs, dtype=torch.bool, device=device
        )
        batch["noise_mask"] = torch.zeros(B, dtype=torch.bool, device=device)
        return batch

    noise_vec = torch.zeros_like(batch["positions"])
    noise_vec = noise_vec.normal_(mean=0.0, std=std)
    dens_batch_mask = torch.ones(num_graphs, device=device, dtype=torch.bool)

    if corrupt_ratio is not None:
        noise_mask = torch.rand(B, dtype=dtype, device=device)
        noise_mask = noise_mask < corrupt_ratio
        noise_vec[(~noise_mask)] *= 0
        batch["noise_mask"] = noise_mask

    # Not add noise to structures with `skip_dens` == True
    if "skip_dens" in batch:
        skip_dens_index = batch["skip_dens"].bool()
        dens_batch_mask = dens_batch_mask * (~skip_dens_index)
        skip_dens_index = skip_dens_index[batch_index]
        noise_mask = ~skip_dens_index
        noise_vec[(~noise_mask)] *= 0
        if "noise_mask" in batch:
            batch["noise_mask"] = batch["noise_mask"] * noise_mask
        else:
            batch["noise_mask"] = noise_mask

    if min_num_atoms is not None:
        noise_mask = num_atoms >= min_num_atoms
        dens_batch_mask = dens_batch_mask * noise_mask
        noise_mask = noise_mask[batch_index]
        noise_vec[(~noise_mask)] *= 0
        if "noise_mask" in batch:
            batch["noise_mask"] = batch["noise_mask"] * noise_mask
        else:
            batch["noise_mask"] = noise_mask

    if strict_max_ratio is not None:
        assert corrupt_ratio is not None
        noise_mask_tensor = batch["noise_mask"].to(dtype=dtype)
        num_corrupted_atoms = torch.zeros(num_graphs, device=device, dtype=dtype)
        num_corrupted_atoms.index_add_(0, batch_index, noise_mask_tensor)
        noise_mask = num_corrupted_atoms <= (num_atoms * strict_max_ratio)
        dens_batch_mask = dens_batch_mask * noise_mask
        noise_mask = noise_mask[batch_index]
        noise_vec[(~noise_mask)] *= 0
        batch["noise_mask"] = batch["noise_mask"] * noise_mask

    if max_forces_norm is not None:
        if "direct_forces" in batch:
            forces_data = batch["direct_forces"]
        else:
            forces_data = batch["forces"]
        forces_norm = torch.norm(forces_data, dim=-1)
        forces_norm_max_reduce = scatter(forces_norm, batch_index, 0, reduce="max")
        noise_mask = forces_norm_max_reduce <= max_forces_norm
        dens_batch_mask = dens_batch_mask * noise_mask
        noise_mask = noise_mask[batch_index]
        noise_vec[(~noise_mask)] *= 0
        if "noise_mask" in batch:
            batch["noise_mask"] = batch["noise_mask"] * noise_mask
        else:
            batch["noise_mask"] = noise_mask

    if max_stress_norm is not None:
        if "direct_stress" in batch:
            stress_data = batch["direct_stress"]
        else:
            stress_data = batch["stress"]
        stress_norm = stress_data.reshape(-1, 9)
        stress_norm = stress_norm**2
        stress_norm = torch.sqrt(torch.sum(stress_norm, dim=1))
        noise_mask = stress_norm <= max_stress_norm
        dens_batch_mask = dens_batch_mask * noise_mask
        noise_mask = noise_mask[batch_index]
        noise_vec[(~noise_mask)] *= 0
        if "noise_mask" in batch:
            batch["noise_mask"] = batch["noise_mask"] * noise_mask
        else:
            batch["noise_mask"] = noise_mask

    if max_mean_forces_norm is not None:
        if "direct_forces" in batch:
            forces_data = batch["direct_forces"]
        else:
            forces_data = batch["forces"]
        forces_reduce = scatter(src=forces_data, index=batch_index, dim=0, reduce="sum")
        forces_reduce_norm = torch.norm(forces_reduce, dim=-1)
        noise_mask = forces_reduce_norm <= max_mean_forces_norm
        dens_batch_mask = dens_batch_mask * noise_mask
        noise_mask = noise_mask[batch_index]
        noise_vec[(~noise_mask)] *= 0
        if "noise_mask" in batch:
            batch["noise_mask"] = batch["noise_mask"] * noise_mask
        else:
            batch["noise_mask"] = noise_mask

    pos = batch["positions"]
    new_pos = pos + noise_vec
    if all_atoms:
        batch["positions"] = new_pos
    else:
        free_mask = batch["fixed"] == 0.0
        batch["positions"][free_mask] = new_pos[free_mask]
    batch["noise_vec"] = noise_vec
    batch["dens_batch_mask"] = dens_batch_mask

    return batch


def _dens_forces_error(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = label["batch"]
    total_weight = (label["entropy"] * label[f"{key}_weight"])[batch]
    noise_mask = label["noise_mask"].unsqueeze(-1)
    forces_error = (pred[key] - label[key]) * (~noise_mask)
    noise_error = (pred["noise_vec"] - label["noise_vec"]) * noise_mask
    return forces_error, noise_error, total_weight


def _validate_dens_loss_ratio(dens_loss_ratio: float) -> None:
    if dens_loss_ratio < 0:
        raise ValueError("dens_loss_ratio must be non-negative.")


def _dens_stress_error(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_key = f"{key}_weight"
    if key == "stress":
        weight_key = "stress_weight"
    total_weight = label["entropy"] * label[weight_key]
    mask = (~label["dens_batch_mask"]).unsqueeze(-1).unsqueeze(-1)
    return (pred[key] - label[key]) * mask, total_weight


def _dens_voigt_stress_error(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    error, total_weight = _dens_stress_error(pred, label, key)
    return voigt6_stress(error), total_weight


@register_loss
def mse_dens_forces(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    dens_loss_ratio: float = 0.05,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    key = "forces"
    _validate_dens_loss_ratio(dens_loss_ratio)
    forces_error, noise_error, total_weight = _dens_forces_error(pred, label, key)
    total_weight = apply_element_weights(total_weight, label, element_weights)
    error = torch.square(forces_error) + dens_loss_ratio * torch.square(noise_error)
    return torch.mean(error * total_weight.unsqueeze(-1))


@register_loss
def mae_dens_forces(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    dens_loss_ratio: float = 0.05,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    key = "forces"
    _validate_dens_loss_ratio(dens_loss_ratio)
    forces_error, noise_error, total_weight = _dens_forces_error(pred, label, key)
    total_weight = apply_element_weights(total_weight, label, element_weights)
    error = torch.abs(forces_error) + dens_loss_ratio * torch.abs(noise_error)
    return torch.mean(error * total_weight.unsqueeze(-1))


@register_loss
def huber_dens_forces(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    dens_loss_ratio: float = 0.05,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    key = "forces"
    _validate_dens_loss_ratio(dens_loss_ratio)
    forces_error, noise_error, total_weight = _dens_forces_error(pred, label, key)
    total_weight = total_weight.unsqueeze(-1)
    forces_error = total_weight * forces_error
    noise_error = total_weight * noise_error
    forces_loss = torch.nn.functional.huber_loss(
        forces_error,
        torch.zeros_like(forces_error),
        reduction="none",
        delta=huber_delta,
    )
    noise_loss = torch.nn.functional.huber_loss(
        noise_error,
        torch.zeros_like(noise_error),
        reduction="none",
        delta=huber_delta,
    )
    loss = forces_loss + dens_loss_ratio * noise_loss
    return torch.mean(apply_element_weights(loss, label, element_weights))


@register_loss
def l2mae_dens_forces(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    dens_loss_ratio: float = 0.05,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    key = "forces"
    _validate_dens_loss_ratio(dens_loss_ratio)
    forces_error, noise_error, total_weight = _dens_forces_error(pred, label, key)
    total_weight = apply_element_weights(total_weight, label, element_weights)
    error = torch.linalg.vector_norm(forces_error, ord=2, dim=-1)
    error = error + dens_loss_ratio * torch.linalg.vector_norm(
        noise_error, ord=2, dim=-1
    )
    return torch.mean(error * total_weight)


@register_loss
def mse_dens_direct_forces(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    dens_loss_ratio: float = 0.05,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    key = "direct_forces"
    _validate_dens_loss_ratio(dens_loss_ratio)
    forces_error, noise_error, total_weight = _dens_forces_error(pred, label, key)
    total_weight = apply_element_weights(total_weight, label, element_weights)
    error = torch.square(forces_error) + dens_loss_ratio * torch.square(noise_error)
    return torch.mean(error * total_weight.unsqueeze(-1))


@register_loss
def mae_dens_direct_forces(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    dens_loss_ratio: float = 0.05,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    key = "direct_forces"
    _validate_dens_loss_ratio(dens_loss_ratio)
    forces_error, noise_error, total_weight = _dens_forces_error(pred, label, key)
    total_weight = apply_element_weights(total_weight, label, element_weights)
    error = torch.abs(forces_error) + dens_loss_ratio * torch.abs(noise_error)
    return torch.mean(error * total_weight.unsqueeze(-1))


@register_loss
def huber_dens_direct_forces(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    huber_delta: float = 0.01,
    dens_loss_ratio: float = 0.05,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    key = "direct_forces"
    _validate_dens_loss_ratio(dens_loss_ratio)
    forces_error, noise_error, total_weight = _dens_forces_error(pred, label, key)
    total_weight = total_weight.unsqueeze(-1)
    forces_error = total_weight * forces_error
    noise_error = total_weight * noise_error
    forces_loss = torch.nn.functional.huber_loss(
        forces_error,
        torch.zeros_like(forces_error),
        reduction="none",
        delta=huber_delta,
    )
    noise_loss = torch.nn.functional.huber_loss(
        noise_error,
        torch.zeros_like(noise_error),
        reduction="none",
        delta=huber_delta,
    )
    loss = forces_loss + dens_loss_ratio * noise_loss
    return torch.mean(apply_element_weights(loss, label, element_weights))


@register_loss
def l2mae_dens_direct_forces(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    dens_loss_ratio: float = 0.05,
    element_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    key = "direct_forces"
    _validate_dens_loss_ratio(dens_loss_ratio)
    forces_error, noise_error, total_weight = _dens_forces_error(pred, label, key)
    total_weight = apply_element_weights(total_weight, label, element_weights)
    error = torch.linalg.vector_norm(forces_error, ord=2, dim=-1)
    error = error + dens_loss_ratio * torch.linalg.vector_norm(
        noise_error, ord=2, dim=-1
    )
    return torch.mean(error * total_weight)


@register_loss
def mse_dens_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "stress"
    error, total_weight = _dens_stress_error(pred, label, key)
    return torch.mean(torch.square(error) * total_weight.unsqueeze(-1).unsqueeze(-1))


@register_loss
def mae_dens_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "stress"
    error, total_weight = _dens_stress_error(pred, label, key)
    return torch.mean(torch.abs(error) * total_weight.unsqueeze(-1).unsqueeze(-1))


@register_loss
def huber_dens_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    key = "stress"
    error, total_weight = _dens_stress_error(pred, label, key)
    weighted_error = total_weight.unsqueeze(-1).unsqueeze(-1) * error
    return torch.nn.functional.huber_loss(
        weighted_error,
        torch.zeros_like(weighted_error),
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def l2mae_dens_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "stress"
    error, total_weight = _dens_stress_error(pred, label, key)
    return torch.mean(torch.linalg.vector_norm(error, ord=2, dim=(1, 2)) * total_weight)


@register_loss
def mse_dens_direct_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_stress"
    error, total_weight = _dens_stress_error(pred, label, key)
    return torch.mean(torch.square(error) * total_weight.unsqueeze(-1).unsqueeze(-1))


@register_loss
def mae_dens_direct_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_stress"
    error, total_weight = _dens_stress_error(pred, label, key)
    return torch.mean(torch.abs(error) * total_weight.unsqueeze(-1).unsqueeze(-1))


@register_loss
def huber_dens_direct_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    key = "direct_stress"
    error, total_weight = _dens_stress_error(pred, label, key)
    weighted_error = total_weight.unsqueeze(-1).unsqueeze(-1) * error
    return torch.nn.functional.huber_loss(
        weighted_error,
        torch.zeros_like(weighted_error),
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def l2mae_dens_direct_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_stress"
    error, total_weight = _dens_stress_error(pred, label, key)
    return torch.mean(torch.linalg.vector_norm(error, ord=2, dim=(1, 2)) * total_weight)


@register_loss
def mse_dens_voigt_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "stress"
    error, total_weight = _dens_voigt_stress_error(pred, label, key)
    return torch.mean(torch.square(error) * total_weight.unsqueeze(-1))


@register_loss
def mae_dens_voigt_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "stress"
    error, total_weight = _dens_voigt_stress_error(pred, label, key)
    return torch.mean(torch.abs(error) * total_weight.unsqueeze(-1))


@register_loss
def huber_dens_voigt_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    key = "stress"
    error, total_weight = _dens_voigt_stress_error(pred, label, key)
    weighted_error = total_weight.unsqueeze(-1) * error
    return torch.nn.functional.huber_loss(
        weighted_error,
        torch.zeros_like(weighted_error),
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def l2mae_dens_voigt_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "stress"
    error, total_weight = _dens_voigt_stress_error(pred, label, key)
    return torch.mean(torch.linalg.vector_norm(error, ord=2, dim=-1) * total_weight)


@register_loss
def mse_dens_voigt_direct_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_stress"
    error, total_weight = _dens_voigt_stress_error(pred, label, key)
    return torch.mean(torch.square(error) * total_weight.unsqueeze(-1))


@register_loss
def mae_dens_voigt_direct_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_stress"
    error, total_weight = _dens_voigt_stress_error(pred, label, key)
    return torch.mean(torch.abs(error) * total_weight.unsqueeze(-1))


@register_loss
def huber_dens_voigt_direct_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    key = "direct_stress"
    error, total_weight = _dens_voigt_stress_error(pred, label, key)
    weighted_error = total_weight.unsqueeze(-1) * error
    return torch.nn.functional.huber_loss(
        weighted_error,
        torch.zeros_like(weighted_error),
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def l2mae_dens_voigt_direct_stress(
    pred: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> torch.Tensor:
    key = "direct_stress"
    error, total_weight = _dens_voigt_stress_error(pred, label, key)
    return torch.mean(torch.linalg.vector_norm(error, ord=2, dim=-1) * total_weight)
